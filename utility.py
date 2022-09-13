import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler


def make_optimizer(args, model):

    trainable = filter(lambda x: x.requires_grad, model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'ADAMax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, optimizer, epoch_steps):

    num_steps = int(args.epochs * epoch_steps)
    warmup_steps = int(args.warmup_epochs * epoch_steps)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        t_mul=1.,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return scheduler


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


class Module_CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=0.001):
        super(Module_CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon ** 2))


def moduleNormalize(frame):
    return torch.cat([(frame[:, 0:1, :, :] - 0.4631),
                      (frame[:, 1:2, :, :] - 0.4352),
                      (frame[:, 2:3, :, :] - 0.3990)], 1)


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def print_and_save(text_str, file_stream):
    print(text_str)
    print(text_str, file=file_stream)
    file_stream.flush()


