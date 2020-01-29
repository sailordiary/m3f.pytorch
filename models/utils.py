import numpy as np
from scipy.signal import medfilt


def concordance_cc2(r1, r2):
     mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
     return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)


def mse(preds, labels):
    return sum((preds - labels) ** 2) / len(labels)


def smooth_predictions(preds, window=7, mode='median'):
    if mode == 'median':
        return np.apply_along_axis(lambda x: medfilt(x, window), 0, preds)
    elif mode == 'wiener':
        return np.apply_along_axis(lambda x: wiener(x, window), 0, preds)


def plot_results(y1, y2, index):
    X = np.arange(len(Y1))
    plt.plot(X, Y1, label="Actual " + index)
    plt.plot(X, Y2, label="Predicted " + index)
    plt.xlabel('Frames') 
    # naming the y axis 
    plt.ylabel(index) 
    plt.title("Aff-Wild2 predictions")
    plt.legend()
    plt.show() 
