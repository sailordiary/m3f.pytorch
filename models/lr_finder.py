# Based on https://github.com/davidtvs/pytorch-lr-finder/blob/master/torch_lr_finder/lr_finder.py
from torch.optim.lr_scheduler import _LRScheduler

from matplotlib import pyplot as plt


class BatchExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(BatchExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def plot_lr(history, skip_start=10, skip_end=5, log_lr=True, show_lr=None):
    """Plots the learning rate range test.
    Arguments:
        skip_start (int, optional): number of batches to trim from the start.
            Default: 10.
        skip_end (int, optional): number of batches to trim from the start.
            Default: 5.
        log_lr (bool, optional): True to plot the learning rate in a logarithmic
            scale; otherwise, plotted in a linear scale. Default: True.
        show_lr (float, optional): is set, will add vertical line to visualize
            specified learning rate; Default: None.
    """
    if skip_start < 0:
        raise ValueError("skip_start cannot be negative")
    if skip_end < 0:
        raise ValueError("skip_end cannot be negative")
    if show_lr is not None and not isinstance(show_lr, float):
        raise ValueError("show_lr must be float")
    # Get the data to plot from the history dictionary. Also, handle skip_end=0
    # properly so the behaviour is the expected
    lrs = history["lr"]
    losses = history["loss"]
    if skip_end == 0:
        lrs = lrs[skip_start:]
        losses = losses[skip_start:]
    else:
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]
    # Plot loss as a function of the learning rate
    print (lrs)
    plt.plot(lrs, losses)
    if log_lr:
        plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    if show_lr is not None:
        plt.axvline(x=show_lr, color="red")
    plt.savefig('lr_plot.png')
