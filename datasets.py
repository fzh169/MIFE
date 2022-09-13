import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class Vimeo90k(Dataset):
    def __init__(self, root, path_list, random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.root = root
        self.path_list = path_list

        self.random_crop = random_crop
        self.resize = resize
        self.augment_s = augment_s
        self.augment_t = augment_t

    def __getitem__(self, index):
        path = self.path_list[index]
        return self.Vimeo90K_loader(path)

    def __len__(self):
        return len(self.path_list)

    def Vimeo90K_loader(self, im_path):
        abs_im_path = os.path.join(self.root, 'sequences', im_path)

        transform_list = []
        if self.resize is not None:
            transform_list += [transforms.Resize(self.resize)]
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

        rawFrame0 = Image.open(os.path.join(abs_im_path, "im1.png"))
        rawFrame1 = Image.open(os.path.join(abs_im_path, "im2.png"))
        rawFrame2 = Image.open(os.path.join(abs_im_path, "im3.png"))

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if random.randint(0, 1):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if random.randint(0, 1):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if random.randint(0, 1):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2


def Vimeo90K_interp(root, list_file, random_crop=None, resize=None, augment_s=True, augment_t=True):

    im_list = open(os.path.join(root, list_file)).read().splitlines()
    im_list = im_list[:-1]  # the last line is invalid in test set
    assert len(im_list) > 0
    random.shuffle(im_list)

    dataset = Vimeo90k(root, im_list, random_crop, resize, augment_s, augment_t)
    return dataset


class SNUFILM(Dataset):
    def __init__(self, root, mode='hard'):

        test_fn = os.path.join(root, 'test-%s.txt' % mode)
        with open(test_fn, 'r') as f:
            self.frame_list = f.read().splitlines()
        self.frame_list = [v.split(' ') for v in self.frame_list]

        for frame in self.frame_list:
            frame[0] = os.path.join(root, frame[0].replace('data/SNU-FILM/', ''))
            frame[1] = os.path.join(root, frame[1].replace('data/SNU-FILM/', ''))
            frame[2] = os.path.join(root, frame[2].replace('data/SNU-FILM/', ''))

        self.transforms = transforms.Compose([transforms.ToTensor()])

        print("[%s] Test dataset has %d triplets" % (mode, len(self.frame_list)))

    def __getitem__(self, index):

        imgpaths = self.frame_list[index]

        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

        img1 = self.transforms(img1)
        img2 = self.transforms(img2)
        img3 = self.transforms(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.frame_list)


def SNUFILM_interp(root, mode="hard"):

    dataset = SNUFILM(root, mode=mode)
    return dataset


class ATD12K(Dataset):
    def __init__(self, root, path_list, random_crop=None, resize=None, augment_s=True, augment_t=True):
        self.root = root
        self.path_list = path_list

        self.random_crop = random_crop
        self.resize = resize
        self.augment_s = augment_s
        self.augment_t = augment_t

    def __getitem__(self, index):
        path = self.path_list[index]
        return self.ATD12K_loader(path)

    def __len__(self):
        return len(self.path_list)

    def ATD12K_loader(self, im_path):

        transform_list = []
        if self.resize is not None:
            transform_list += [transforms.Resize(self.resize)]
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

        rawFrame0 = Image.open(os.path.join(im_path, "frame1.png"))
        rawFrame1 = Image.open(os.path.join(im_path, "frame2.png"))
        rawFrame2 = Image.open(os.path.join(im_path, "frame3.png"))

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(rawFrame1, output_size=self.random_crop)
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if random.randint(0, 1):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if random.randint(0, 1):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if random.randint(0, 1):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2


def ATD12K_interp(root, random_crop=None, resize=None, augment_s=True, augment_t=True):

    im_list = []
    for index, folder in enumerate(os.listdir(root)):
        path = os.path.join(root, folder)
        if (os.path.isdir(path)) is not None:
            im_list.append(path)

    dataset = ATD12K(root, im_list, random_crop, resize, augment_s, augment_t)
    return dataset
