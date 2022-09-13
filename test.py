import numpy as np
import torch
import cv2
from torchvision.utils import save_image as imwrite
from torch.utils.data import DataLoader

import math
from datasets import *
from utility import print_and_save, count_network_parameters
from pytorch_msssim import ssim_matlab
from vis_flow import flow_to_color
from skimage.color import rgb2yuv, yuv2rgb
from yuv_frame_io import YUV_Read, YUV_Write
import os
import time


def save_flow_to_img(flow, des):
    f = flow[0].data.cpu().numpy().transpose([1, 2, 0])
    fcopy = f.copy()
    fcopy[:, :, 0] = f[:, :, 1]
    fcopy[:, :, 1] = f[:, :, 0]
    cf = flow_to_color(-fcopy)
    cv2.imwrite(des + '.png', cf)


class Middlebury_other:
    def __init__(self, input_dir, gt_dir):

        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.list = ['snu1', 'Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale',
                     'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

    def test(self, model, output_dir, output_name='output', file_stream=None):

        model.eval()
        with torch.no_grad():
            av_psnr = 0
            av_ssim = 0
            av_IE = 0
            print('%25s%21s%21s' % ('PSNR', 'SSIM', 'IE'))
            for i in self.list:
                img0 = (self.input_dir + '/{}/frame10.png'.format(i))
                img1 = (self.input_dir + '/{}/frame11.png'.format(i))
                gt = (self.gt_dir + '/{}/frame10i11.png'.format(i))

                img0 = self.transform(Image.open(img0)).cuda().unsqueeze(0)
                img1 = self.transform(Image.open(img1)).cuda().unsqueeze(0)
                gt = self.transform(Image.open(gt)).cuda().unsqueeze(0)

                t0 = time.time()
                pred = model(img0, img1)[0]
                t1 = time.time()
                print('miao: ', t1 - t0)
                ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

                out = pred.detach().cpu().numpy().transpose(1, 2, 0)
                out = np.round(out * 255) / 255.
                gt = gt[0].cpu().numpy().transpose(1, 2, 0)
                psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())

                out = pred.detach().cpu().numpy().transpose(1, 2, 0)
                out = np.round(out * 255)
                gt = np.round(gt * 255) * 1.0
                IE = np.abs((out - gt)).mean()

                av_psnr += psnr
                av_ssim += ssim
                av_IE += IE

                if not os.path.exists(output_dir + '/' + i):
                    os.makedirs(output_dir + '/' + i)
                imwrite(pred, output_dir + '/' + i + '/' + output_name + '.png', range=(0, 1))

                msg = '{:<15s}{:<23.16f}{:<23.16f}{:<23.16f}'.format(i + ': ', psnr, ssim, IE)
                if file_stream:
                    print_and_save(msg, file_stream)
                else:
                    print(msg)

        av_psnr /= len(self.list)
        av_ssim /= len(self.list)
        av_IE /= len(self.list)

        msg = '\n{:<15s}{:<23.16f}{:<23.16f}{:<23.16f}'.format('Average: ', av_psnr, av_ssim, av_IE)
        if file_stream:
            print_and_save(msg, file_stream)
        else:
            print(msg)

        return av_psnr

    def save_flow(self, model, output_dir, output_name='output'):

        model.eval()
        with torch.no_grad():
            for i in self.list:
                img0 = (self.input_dir + '/{}/frame10.png'.format(i))
                img1 = (self.input_dir + '/{}/frame11.png'.format(i))

                img0 = self.transform(Image.open(img0)).cuda().unsqueeze(0)
                img1 = self.transform(Image.open(img1)).cuda().unsqueeze(0)

                pred = model(img0, img1)

                if not os.path.exists(output_dir + '/' + i):
                    os.makedirs(output_dir + '/' + i)
                imwrite(pred[0], output_dir + '/' + i + '/' + output_name + '.png', range=(0, 1))

                save_flow_to_img(pred[1].cpu(), output_dir + '/' + i + '_F10')
                save_flow_to_img(pred[2].cpu(), output_dir + '/' + i + '_F12')


class Middlebury_eval:
    def __init__(self, input_dir):

        self.input_dir = input_dir
        self.list = ['Army', 'Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Grove', 'Mequon', 'Schefflera',
                     'Teddy', 'Urban', 'Wooden', 'Yosemite']
        self.transform = transforms.Compose([transforms.ToTensor()])

    def test(self, model, output_dir, output_name='output'):

        model.eval()
        with torch.no_grad():
            for i in self.list:
                img0 = (self.input_dir + '/{}/frame10.png'.format(i))
                img1 = (self.input_dir + '/{}/frame11.png'.format(i))

                img0 = self.transform(Image.open(img0)).cuda().unsqueeze(0)
                img1 = self.transform(Image.open(img1)).cuda().unsqueeze(0)

                pred = model(img0, img1)[0]

                if not os.path.exists(output_dir + '/' + i):
                    os.makedirs(output_dir + '/' + i)
                imwrite(pred, output_dir + '/' + i + '/' + output_name + '.png', range=(0, 1))


class ucf_dvf:
    def __init__(self, input_dir):

        self.path = input_dir
        self.list = [str(x) for x in list(range(1, 3791, 10))]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def test(self, model, output_dir, output_name='output', file_stream=None):

        model.eval()
        with torch.no_grad():
            av_psnr = 0
            av_ssim = 0
            print('%25s%21s' % ('PSNR', 'SSIM'))
            for i in self.list:
                img0 = (self.path + '/' + i + '/frame_00.png')
                img1 = (self.path + '/' + i + '/frame_02.png')
                gt = (self.path + '/' + i + '/frame_01_gt.png')

                img0 = self.transform(Image.open(img0)).cuda().unsqueeze(0)
                img1 = self.transform(Image.open(img1)).cuda().unsqueeze(0)
                gt = self.transform(Image.open(gt)).cuda().unsqueeze(0)

                pred = model(img0, img1)[0]
                ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

                out = pred.detach().cpu().numpy().transpose(1, 2, 0)
                out = np.round(out * 255) / 255.
                gt = gt[0].cpu().numpy().transpose(1, 2, 0)
                psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())

                av_psnr += psnr
                av_ssim += ssim

                if not os.path.exists(output_dir + '/' + i):
                    os.makedirs(output_dir + '/' + i)
                imwrite(pred, output_dir + '/' + i + '/' + output_name + '.png', range=(0, 1))

                msg = '{:<15s}{:<23.16f}{:<23.16f}'.format(i + ': ', psnr, ssim)
                if file_stream:
                    print_and_save(msg, file_stream)
                else:
                    print(msg)

        av_psnr /= len(self.list)
        av_ssim /= len(self.list)

        msg = '\n{:<15s}{:<23.16f}{:<23.16f}'.format('Average: ', av_psnr, av_ssim)
        if file_stream:
            print_and_save(msg, file_stream)
        else:
            print(msg)


class HD:
    def __init__(self, input_dir):

        self.input_dir = input_dir
        self.list = [
            ('HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
            ('HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
            ('HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
            ('HD1080p_GT/BlueSky.yuv', 1080, 1920),
            ('HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
            ('HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
            ('HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
            ('HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
            ('HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
            ('HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
            ('HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280)
        ]

    def test(self, model, out_dir):

        model.eval()
        with torch.no_grad():
            av_psnr = 0
            for data in self.list:
                name, h, w = data[0], data[1], data[2]
                Reader = YUV_Read(self.input_dir + '/' + name, h, w, toRGB=True)
                _, lastframe = Reader.read()

                psnr, count = 0, 0
                for index in range(0, 100, 2):
                    IMAGE1, success1 = Reader.read(index)
                    gt, _ = Reader.read(index + 1)
                    IMAGE2, success2 = Reader.read(index + 2)
                    if not success2:
                        break

                    I0 = torch.from_numpy(np.transpose(IMAGE1, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)
                    I1 = torch.from_numpy(np.transpose(IMAGE2, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)

                    pred = model(I0, I1)
                    out = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')

                    diff_rgb = 128.0 + rgb2yuv(gt / 255.)[:, :, 0] * 255 - rgb2yuv(out / 255.)[:, :, 0] * 255
                    mse = np.mean((diff_rgb - 128.0) ** 2)
                    PIXEL_MAX = 255.0
                    psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
                    count += 1

                    if not os.path.exists(out_dir + '/' + name):
                        os.makedirs(out_dir + '/' + name)
                    imwrite(pred, out_dir + '/' + name + '.png', range=(0, 1))

                psnr /= count
                av_psnr += psnr
                print(name + ': ', psnr)

            av_psnr /= len(self.list)
            print('\n{:<15s}{:<23.16f}'.format('Average: ', av_psnr))


def Vimeo90K_test(model, vimeo_dir, out_dir):
    val_dataset = Vimeo90K_interp(vimeo_dir, "tri_testlist.txt", random_crop=None, resize=None,
                                  augment_s=False, augment_t=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1)

    img_out_dir = out_dir + '/vimeo90k'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    av_psnr = 0
    av_ssim = 0
    for batch_idx, (frame0, frame1, frame2) in enumerate(val_loader):
        frame0, frame1, frame2 = frame0.cuda(), frame1.cuda(), frame2.cuda()

        pred = model(frame0, frame2)[0]
        ssim = ssim_matlab(frame1, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

        out = pred.detach().cpu().numpy().transpose(1, 2, 0)
        out = np.round(out * 255) / 255.
        gt = frame1[0].cpu().numpy().transpose(1, 2, 0)
        psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())

        av_psnr += psnr
        av_ssim += ssim

        imwrite(pred, img_out_dir + '/' + str(batch_idx) + '.png', range=(0, 1))

        print('{:<10s}{:<23.16f}{:<23.16f}'.format(str(batch_idx) + ': ', psnr, ssim))

    av_psnr /= len(val_loader)
    av_ssim /= len(val_loader)

    print('\n{:<10s}{:<23.16f}{:<23.16f}'.format('Average: ', av_psnr, av_ssim))


def SNUFILM_test(model, snu_dir, out_dir, mode):

    val_dataset = SNUFILM_interp(snu_dir, mode)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1)

    img_out_dir = out_dir + '/snufilm'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    av_psnr = 0
    av_ssim = 0
    for batch_idx, (frame0, frame1, frame2) in enumerate(val_loader):
        frame0, frame1, frame2 = frame0.cuda(), frame1.cuda(), frame2.cuda()

        pred = model(frame0, frame2)[0]
        ssim = ssim_matlab(frame1, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

        out = pred.detach().cpu().numpy().transpose(1, 2, 0)
        out = np.round(out * 255) / 255.
        gt = frame1[0].cpu().numpy().transpose(1, 2, 0)
        psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())

        av_psnr += psnr
        av_ssim += ssim

        imwrite(pred, img_out_dir + '/' + str(batch_idx) + '.png', range=(0, 1))

        print('{:<10s}{:<23.16f}{:<23.16f}'.format(str(batch_idx) + ': ', psnr, ssim))

    av_psnr /= len(val_loader)
    av_ssim /= len(val_loader)

    print('\n{:<10s}{:<23.16f}{:<23.16f}'.format('Average: ', av_psnr, av_ssim))


def ATD12K_test(model, atd_dir, out_dir):
    val_dataset = ATD12K_interp(atd_dir, random_crop=None, resize=None, augment_s=False, augment_t=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1)

    img_out_dir = out_dir + '/atd12k'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    av_psnr = 0
    av_ssim = 0
    for batch_idx, (frame0, frame1, frame2) in enumerate(val_loader):
        frame0, frame1, frame2 = frame0.cuda(), frame1.cuda(), frame2.cuda()

        pred = model(frame0, frame2)[0]
        ssim = ssim_matlab(frame1, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

        out = pred.detach().cpu().numpy().transpose(1, 2, 0)
        out = np.round(out * 255) / 255.
        gt = frame1[0].cpu().numpy().transpose(1, 2, 0)
        psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())

        av_psnr += psnr
        av_ssim += ssim

        imwrite(pred, img_out_dir + '/' + str(batch_idx) + '.png', range=(0, 1))

        print('{:<10s}{:<23.16f}{:<23.16f}'.format(str(batch_idx) + ': ', psnr, ssim))

    av_psnr /= len(val_loader)
    av_ssim /= len(val_loader)

    print('\n{:<10s}{:<23.16f}{:<23.16f}'.format('Average: ', av_psnr, av_ssim))


def test(model):

    print('===============================')
    print("# of model parameters is: " + str(count_network_parameters(model)))

    model.eval()

    print('===============================')
    print('Test: Middlebury_others')
    img_out_dir = './test_output/middlebury_others'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    test_db = Middlebury_other('./test_data/middlebury_others/input', './test_data/middlebury_others/gt')
    test_db.test(model, img_out_dir)
    # test_db.save_flow(model, img_out_dir)

    # print('===============================')
    # print('Test: Middlebury_eval')
    # img_out_dir = './test_output/middlebury_eval'
    # if not os.path.exists(img_out_dir):
    #     os.makedirs(img_out_dir)
    # test_db = Middlebury_eval('./test_data/eval-color-twoframes/eval-data')
    # test_db.test(model, img_out_dir)

    # print('===============================')
    # print('Test: UCF101-DVF')
    # img_out_dir = './test_output/ucf101-dvf'
    # if not os.path.exists(img_out_dir):
    #     os.makedirs(img_out_dir)
    # test_db = ucf_dvf('./test_data/ucf101_interp')
    # test_db.test(model, img_out_dir)
    #
    # print('===============================')
    # print('Test: HD_dataset')
    # img_out_dir = './test_output/HD_dataset'
    # if not os.path.exists(img_out_dir):
    #     os.makedirs(img_out_dir)
    # test_db = HD('./test_data/HD_dataset')
    # test_db.test(model, img_out_dir)
    #
    # print('===============================')
    # print('Test: Vimeo-90K')
    # with torch.no_grad():
    #     Vimeo90K_test(model, './train_data/vimeo_triplet/', './test_output')
    #
    # for mode in ["easy", "medium", "hard", "extreme"]:
    #     print('===============================')
    #     print('Test: SNU-FILM    mode: ' + mode)
    #     with torch.no_grad():
    #         SNUFILM_test(model, './test_data/snufilm', './test_output', mode)
    #
    # print('===============================')
    # print('Test: ATD-12K')
    # with torch.no_grad():
    #     ATD12K_test(model, './test_data/atd_12k/test_2k_original/', './test_output')
