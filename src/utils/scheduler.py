import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0.
        if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def get_scheduler(
        config,
        optimizer: Optimizer,
) -> _LRScheduler:
    """
    The function receives config and optimizer and returns lr_scheduler.

    Parameters
    ----------
    config
        Config.
    optimizer: Optimizer
        Optimizer.

    Returns
    -------
    scheduler: _LRScheduler
        Learning rate scheduler.

    """
    if config.scheduler_policy == 'cos_an_rest':
        if config.num_restarts < 0:
            raise ValueError("num_restarts cannot be negative.")
        else:
            total_epochs = config.total_epochs - config.warmup_epoch_num
            assert total_epochs > 0, 'Invalid epoch num!'

            T_0 = math.ceil(total_epochs * config.num_samples_per_gpu / (config.num_restarts + 1))
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 + 1)

    elif config.scheduler_policy == 'step_mr':
        new_milestones = [mile * config.num_samples_per_gpu for mile in config.lr_milestones]

        scheduler = MultiStepLR(
            optimizer,
            new_milestones,
            gamma=config.lr_milestone_gamma,
        )
    else:
        raise NotImplementedError(f"This type of scheduler is not supported.\nGot: '{config.scheduler}'")

    if config.warmup_epoch_num > 0:
        scheduler_with_warmup = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config.warmup_epoch_num * config.num_samples_per_gpu,
            after_scheduler=scheduler,
        )
    else:
        scheduler_with_warmup = scheduler

    return scheduler_with_warmup
