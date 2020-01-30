import numpy as np
from scipy.signal import medfilt, wiener
from matplotlib import pyplot as plt


def concordance_cc2(r1, r2, reduction='mean'):
    '''
    Computes batch element-wise CCC.
    '''
    r1_mean = r1.mean(dim=-1, keepdims=True)
    r2_mean = r2.mean(dim=-1, keepdims=True)
    mean_cent_prod = ((r1 - r1_mean * (r2 - r2_mean))).mean(dim=-1, keepdims=True)
    ccc = (2 * mean_cent_prod) / (r1.var(dim=-1, keepdims=True) + r2.var(dim=-1, keepdims=True) + (r1_mean - r2_mean) ** 2)
    if reduction == 'none':
        return ccc
    elif reduction == 'mean':
        return ccc.mean()


def mse(preds, labels):
    return sum((preds - labels) ** 2) / len(labels)


def smooth_predictions(preds, window=7, mode='wiener'):
    if mode == 'median':
        return np.apply_along_axis(lambda x: medfilt(x, window), 0, preds)
    elif mode == 'wiener':
        return np.apply_along_axis(lambda x: wiener(x, window), 0, preds)


def plot_results(y1, y2, index):
    X = np.arange(len(y1))
    plt.plot(X, y1, label="Actual " + index)
    plt.plot(X, y2, label="Predicted " + index)
    plt.xlabel('Frames') 
    # naming the y axis 
    plt.ylabel(index) 
    plt.title("Aff-Wild2 predictions")
    plt.legend()
    plt.show() 
