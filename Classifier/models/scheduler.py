# scheduler from noise2sim repo

import torch
import numpy as np

class RampedLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, iteration_count, ramp_up_fraction, ramp_down_fraction, last_epoch=-1):
        self.iteration_count = iteration_count
        self.ramp_up_fraction = ramp_up_fraction
        self.ramp_down_fraction = ramp_down_fraction

        if ramp_up_fraction > 0.0:
            self.ramp_up_end_iter = iteration_count * ramp_up_fraction
        else:
            self.ramp_up_end_iter = None

        if ramp_down_fraction > 0.0:
            self.ramp_down_start_iter = iteration_count * (1 - ramp_down_fraction)
        else:
            self.ramp_down_start_iter = None

        super(RampedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.ramp_up_end_iter is not None:
            if self.last_epoch <= self.ramp_up_end_iter:
                return [base_lr * (0.5 - np.cos(((self.last_epoch / self.ramp_up_fraction) / self.iteration_count) * np.pi)/2)
                        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)]

        if self.ramp_down_fraction is not None:
            if self.last_epoch >= self.ramp_down_start_iter:
                return [base_lr * (0.5 + np.cos((((self.last_epoch - self.ramp_down_start_iter) / self.ramp_down_fraction) / self.iteration_count) * np.pi)/2)**2
                        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)]

        return [group['lr'] for group in self.optimizer.param_groups]