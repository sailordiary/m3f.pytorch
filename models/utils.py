import numpy as np
from scipy.signal import medfilt, wiener
from matplotlib import pyplot as plt

'''
import pyaudio
import wave


class AudioPlayer:
    """
    Player implemented with PyAudio

    http://people.csail.mit.edu/hubert/pyaudio/

    Mac OS X:

      brew install portaudio
      pip install http://people.csail.mit.edu/hubert/pyaudio/packages/pyaudio-0.2.8.tar.gz
    """
    def __init__(self, wav):
        self.p = pyaudio.PyAudio()
        self.pos = 0
        self.stream = None
        self._open(wav)

    def callback(self, in_data, frame_count, time_info, status):
        data = self.wf.readframes(frame_count)
        self.pos += frame_count
        return (data, pyaudio.paContinue)

    def _open(self, wav):
        self.wf = wave.open(wav, 'rb')
        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                channels = self.wf.getnchannels(),
                rate = self.wf.getframerate(),
                output=True,
                stream_callback=self.callback)
        self.pause()

    def play(self):
        self.stream.start_stream()

    def pause(self):
        self.stream.stop_stream()

    def seek(self, seconds = 0.0):
        sec = seconds * self.wf.getframerate()
        self.pos = int(sec)
        self.wf.setpos(int(sec))

    def time(self):
        return float(self.pos)/self.wf.getframerate()

    def playing(self):
        return self.stream.is_active()

    def close(self):
        self.stream.close()
        self.wf.close()
        self.p.terminate()
'''


def concordance_cc2(r1, r2, reduction='mean'):
    '''
    Computes row-wise CCC.
    '''
    r1_mean = r1.mean(dim=-1, keepdim=True)
    r2_mean = r2.mean(dim=-1, keepdim=True)
    mean_cent_prod = ((r1 - r1_mean) * (r2 - r2_mean)).mean(dim=-1, keepdim=True)
    ccc = (2 * mean_cent_prod) / (r1.var(dim=-1, keepdim=True) + r2.var(dim=-1, keepdim=True) + (r1_mean - r2_mean) ** 2)
    if reduction == 'none':
        return ccc
    elif reduction == 'mean':
        return ccc.mean()


def concordance_cc2_np(r1, r2):
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)


def mse(preds, labels):
    return sum((preds - labels) ** 2) / len(labels)


def smooth_predictions(preds, window=7, mode='wiener'):
    if mode == 'median':
        return np.apply_along_axis(lambda x: medfilt(x, window), 0, preds)
    elif mode == 'wiener':
        return np.apply_along_axis(lambda x: wiener(x, window), 0, preds)


def plot_results(base_path, y1, y2, index):
    X = np.arange(len(y1))
    plt.plot(X, y1, label="Actual " + index)
    plt.plot(X, y2, label="Predicted " + index)
    plt.xlabel('Frames') 
    # naming the y axis 
    plt.ylabel(index) 
    plt.title("Aff-Wild2 predictions")
    plt.legend()
    plt.show() 
