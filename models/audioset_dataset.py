import os

import math
import random

import numpy as np
import librosa

import torch
from torch.utils.data import Dataset


FPS_VALUES = [15.0, 17.0, 19.0, 22.0, 23.976, 24.0, 25.0, 29.97, 30.0]


# Modified from https://github.com/DemisEom/SpecAugment/blob/master/SpecAugment/spec_augment_pytorch.py
def spec_augment(mel_spectrogram, frequency_masking_para=20,
                 time_masking_para=20, frequency_mask_num=1, time_mask_num=1):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): masked mel spectrogram.
    """
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[2]

    # XXX(yuanhang): removed time warping since it makes little sense here

    # Step 1 : Frequency masking
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, v-f)
        mel_spectrogram[:, f0:f0+f, :] = 0

    # Step 2 : Time masking
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, tau-t)
        mel_spectrogram[:, :, t0:t0+t] = 0

    return mel_spectrogram


def load_audio(path, length, is_training=False):
    # test at 30 fps
    fps = random.choice(FPS_VALUES) if is_training else 30.0
    y, sr = librosa.load(path, sr=16000)
    # audio duration: temporal crop
    nsamples = int(length / fps * 16000)
    tot_samples = len(y)
    start = random.randint(0, tot_samples - nsamples) if is_training else (tot_samples - nsamples) // 2
    y = y[start: start + nsamples]
    # win_length = 0.025 * 16000
    hop_length = int(1/3 * 1/fps * 16000)
    power = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512,
                                           hop_length=hop_length, win_length=400, 
                                           n_mels=40)
    spec = librosa.core.power_to_db(power).transpose() # time, channel
    stacked_features = []
    for i in range(length):
        # context_width = 2
        window_feats = spec[i * 3: i * 3 + 5]
        nframes = len(window_feats)
        if len(window_feats) < 5:
            window_feats = np.pad(window_feats, ((0, 5-nframes), (0, 0)), 'constant')
        stacked_features.append(window_feats.reshape(-1))
    stacked_features = np.stack(stacked_features)

    return stacked_features


class AudioSetDataset(Dataset):
    '''
    AudioSet stacked log-mels.
    
    Params:
    split: partition (train, val)
    path: base path for data
    window_len: length of temporal crop window
    '''
    def __init__(self, split, path, window_len=32):
        self.split = split
        self.path = path
        self.window_len = window_len

        fold_map = {'train': 'balanced_train_segments', 'val': 'eval_segments'}
        annot_path = os.path.join(self.path, '{}.csv'.format(fold_map[self.split]))
        lines = open(os.path.join(self.path, 'class_labels_indices.csv'), 'r').read().splitlines()[1: ]
        mid_map = {l.split(',')[1]: i for i, l in enumerate(lines)}

        self.files = []
        for l in open(annot_path, 'r').read().splitlines()[3: ]:
            name, _, _, ontology = l.split(', ')
            wav_path = os.path.join(self.path, fold_map[self.split], name + '.wav')
            if not os.path.exists(wav_path): continue
            class_labels = ontology[1: -1].split(',')
            # multi-class training
            label = np.zeros(527, dtype=np.float32)
            for mid in class_labels:
                label[mid_map[mid]] = 1.
            self.files.append((wav_path, label))
        
        print ('Loaded partition {}: {} samples'.format(self.split, len(self.files)))

    def __getitem__(self, i):
        aud_name = self.files[i][0]
        track_len = self.window_len
        is_training = self.split == 'train'
        inputs = load_audio(aud_name, track_len, is_training)
        
        return {
            'audio': torch.from_numpy(inputs),
            'label': torch.from_numpy(self.files[i][1])
        }

    def __len__(self):
        return len(self.files)
