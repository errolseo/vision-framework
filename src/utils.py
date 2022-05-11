
import random

import torch
import numpy as np


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmt_str = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt_str.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmt_str = self._get_batch_fmt_str(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, accelerator, batch):
        entries = [self.prefix + self.batch_fmt_str.format(batch)]
        entries += [str(meter) for meter in self.meters]
        accelerator.print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmt_str(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'