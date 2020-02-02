import os

import math
import random

from tqdm import tqdm

import numpy as np
import cv2
import pickle

import torch
from torch.utils.data import Dataset


def sequence_cutout(seq, n_holes=1, fill_value=127.5):
    h = seq.shape[-2]
    w = seq.shape[-1]
    length = h // 4

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - length, 0, h)
        y2 = np.clip(y + length, 0, h)
        x1 = np.clip(x - length, 0, w)
        x2 = np.clip(x + length, 0, w)
        # this will be zero after normalisation
        seq[:, :, y1: y2, x1: x2] = fill_value

    return seq


# find consecutive "True"s in a 1D array
# https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array/24892274#24892274
def one_runs(a):
    # Create an array that is 1 where a is 1, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    return ranges


def load_video(path, start, length,
               is_training=False,
               mirror_augment=False,
               crop_augment=False,
               cutout_augment=False,
               input_size=256):
    frames = []
    # use identical crop windows for every frame
    if crop_augment:
        if is_training:
            crop_x = random.randint(0, input_size // 8)
            crop_y = random.randint(0, input_size // 8)
        else:
            crop_x, crop_y = input_size // 16, input_size // 16
        crop_size = input_size * 7 // 8
        resize = input_size > 128

    for i in range(start, start + length):
        img = cv2.imread(os.path.join(path, '{:05d}.jpg'.format(i + 1)))
        # fill in the missing frames with a previous frame or zeros
        if img is None:
            if len(frames) > 0: img = frames[-1]
            else: img = np.zeros((112, 112, 3), dtype=np.uint8)
        # image is present
        else:
            if crop_augment:
                img = img[crop_y: crop_y + crop_size, crop_x: crop_x + crop_size]
                if resize: img = cv2.resize(img, (112, 112))
            if mirror_augment and is_training: img = cv2.flip(img, 1)
        # TODO: add temporal augmentation (repeat, deletion)
        frames.append(img)
    seq = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) # THWC->CTHW
    if cutout_augment and is_training:
        seq = sequence_cutout(seq)
    return seq


def load_audio(audio_path, start_idx, w_len):
    mel_spec = np.load(audio_path)
    stacked_features = []
    for i in range(w_len):
        # context_width = 2
        window_feats = mel_spec[(start_idx + i) * 3: (start_idx + i) * 3 + 5]
        nframes = len(window_feats)
        if len(window_feats) < 5:
            window_feats = np.pad(window_feats, ((0, 5-nframes), (0, 0)), 'constant')
        stacked_features.append(window_feats.reshape(-1))
    stacked_features = np.stack(stacked_features)

    return stacked_features


class AffWild2SequenceDataset(Dataset):
    '''
    112 by 112 face tracks using Aff-Wild2 crops.
    
    Params:
    split: partition (train, test, val)
    path: base path for data
    window_len: length of temporal crop window
    windows_per_epoch: how many times each video should be sampled in one epoch
    apply_cutout: use Cutout augmentation
    release: 'ibug' -- 112*112 ArcFace crops; 'vipl' -- (256*256->)128*128->112*112 VIPL crops
    input_size: actual size of raw input images
    '''
    def __init__(self, split, path, window_len=16, windows_per_epoch=20, apply_cutout=True, release='ibug', input_size=112):
        self.split = split
        self.path = path
        self.window_len = window_len
        self.windows_per_epoch = windows_per_epoch
        self.apply_cutout = apply_cutout
        self.release = release
        self.input_size = input_size
        
        self.base = os.path.join(self.path, 'cropped_aligned' if self.release == 'ibug' else 'face_{}'.format(self.input_size))
        self.nb_frames = {}
        self.fps = {}
        for l in open('splits/frames_fps.csv', 'r').read().splitlines():
            name, nframes, fps = l.split(',')
            self.nb_frames[name] = int(nframes)
            self.fps[name] = float(fps)
        
        self.files = open('splits/{}.csv'.format(self.split), 'r').read().splitlines()
        if self.split == 'train':
            num_files = len(self.files)
            # indices of video to sample from
            self.sample_src = list(range(num_files)) * self.windows_per_epoch
            random.shuffle(self.sample_src)
        else:
            # non-overlapped inference (stride=window_len)
            self.sample_src = []
            for i, vid_name in enumerate(self.files):
                # pairs of (video_idx, window_start)
                for j in range(self.nb_frames[vid_name] // self.window_len):
                    self.sample_src.append((i, j * self.window_len))
        if self.split != 'test':
            self.labels = {}
            for vid_name in self.files:
                # valence, arousal
                lines = open(os.path.join(self.path, 'annotation', self.split, vid_name + '.txt'), 'r').read().splitlines()
                self.labels[vid_name] = np.loadtxt(lines, delimiter=',', skiprows=1, dtype=np.float32)
        if self.split == 'train':
            self.avail_windows = self.get_available_windows()
        
        print ('Loaded partition {}: {} files, {} windows'.format(self.split, len(self.files), len(self.sample_src)))
    
    def get_available_windows(self):
        windows = {k: [] for k in self.files}
        cache_path = '{}_{}_window{}.pkl'.format(self.release, self.split, self.window_len)
        if os.path.exists(cache_path):
            return pickle.load(open(cache_path, 'rb'))
        for vid_name in tqdm(self.files, desc='Scanning available windows'):
            src_fold = os.path.join(self.base, vid_name)
            has_image = np.array([os.path.exists(os.path.join(src_fold, '{:05d}.jpg'.format(i + 1))) for i in range(len(self.labels[vid_name]))])
            has_label = np.max(np.abs(self.labels[vid_name]), axis=1) <= 1
            avail_ranges = one_runs(has_image & has_label)
            for w_st, w_ed in avail_ranges:
                windows[vid_name].extend(list(range(w_st, w_ed - self.window_len + 1)))
            # we're up all night to get lucky :(
            assert len(windows[vid_name]) > 0, 'no available windows for {}'.format(vid_name)
        pickle.dump(windows, open(cache_path, 'wb'))
        return windows

    def __getitem__(self, i):
        if self.split == 'train':
            vid_idx = self.sample_src[i]
            vid_name = self.files[vid_idx]
            track_len = self.window_len
            start_frame = random.choice(self.avail_windows[vid_name])
        else:
            vid_idx, start_frame = self.sample_src[i]
            vid_name = self.files[vid_idx]
            track_len = min(self.window_len, self.nb_frames[vid_name] - start_frame)
        
        # note that frame indices begin with 1
        is_training = self.split == 'train'
        src_vid_fold = os.path.join(self.base, vid_name)
        inputs = load_video(src_vid_fold, start_frame, track_len,
                            self.split == 'train',
                            random.random() > 0.5,
                            self.release == 'vipl',
                            self.apply_cutout,
                            self.input_size)
        
        if self.fps[vid_name] < 15:
            audio = np.zeros((self.window_len, 40), dtype=np.float32)
        else:
            src_aud_fold = os.path.join(self.path, 'mel_spec', vid_name + '.npy')
            audio = load_audio(src_aud_fold, start_frame, track_len)
        
        if self.split != 'test':
            labels = self.labels[vid_name][start_frame: start_frame + track_len]
        # pad with boundary values, which will be discarded for evaluation
        to_pad = self.window_len - track_len
        if to_pad != 0:
            inputs = np.pad(inputs, ((0, 0), (0, to_pad), (0, 0), (0, 0)), 'edge')
            if self.split != 'test':
                labels = np.pad(labels, ((0, to_pad), (0, 0)), 'edge')
        batch = {
            'video': torch.from_numpy(inputs),
            'vid_name': vid_name,
            'start': start_frame,
            'length': track_len
        }
        if self.split != 'test':
            batch['label_valence'] = torch.from_numpy(labels[..., 0])
            # discretize valence into categories
            bins = np.linspace(-1, 1, 4, endpoint=False)
            class_labels = np.digitize(labels[..., 0], bins) - 1
            batch['class_valence'] = torch.from_numpy(class_labels)
            batch['label_arousal'] = torch.from_numpy(labels[..., 1])
        
        return batch

    def __len__(self):
        return len(self.sample_src)
