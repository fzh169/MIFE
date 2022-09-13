import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F

from utility import Module_CharbonnierLoss


class VGG_loss(nn.Module):
    def __init__(self):
        super(VGG_loss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'rec':
                loss_function = Module_CharbonnierLoss()
            elif loss_type.find('vgg') >= 0:
                loss_function = VGG_loss()
            elif loss_type == 'occ':
                loss_function = Module_CharbonnierLoss()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        for l in self.loss:
            if l['function'] is not None:
                self.loss_module.append(l['function'])

        self.loss_module.to('cuda')

    def forward(self, output, input_frames):
        losses = []
        for l in self.loss:
            if l['function'] is not None:
                if l['type'] == 'rec':
                    loss1 = l['function'](output['frame1'], input_frames[1])
                    loss10 = l['function'](output['f10'], input_frames[1])
                    loss12 = l['function'](output['f12'], input_frames[1])
                    lossg0 = l['function'](output['g0'], input_frames[1])
                    lossg1 = l['function'](output['g1'], input_frames[1])
                    lossg2 = l['function'](output['g2'], input_frames[1])
                    loss = loss1 + 0.1 * (loss10 + loss12) + 0.1 * (lossg0 + lossg1 + lossg2)
                elif l['type'] == 'vgg':
                    loss = l['function'](output['frame1'], input_frames[1])
                elif l['type'] == 'occ':
                    lossf0 = l['function'](output['f0'], input_frames[0])
                    lossf2 = l['function'](output['f2'], input_frames[2])
                    loss = lossf0 + lossf2
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum
