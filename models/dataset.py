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
    def __init__(self, split, path, window_len=16, windows_per_epoch=20, apply_cutout=True, release='ibug', input_size=112, modality='visual', noise_and_balance=False):
        self.split = split
        self.path = path
        self.window_len = window_len
        self.windows_per_epoch = windows_per_epoch
        self.apply_cutout = apply_cutout
        self.release = release
        self.input_size = input_size
        self.modality = modality
        
        self.base = os.path.join(self.path, 'cropped_aligned' if self.release == 'ibug' else 'face_{}'.format(self.input_size))
        self.nb_frames = {}
        self.fps = {}
        for l in open('splits/frames_fps.csv', 'r').read().splitlines():
            name, nframes, fps = l.split(',')
            self.nb_frames[name] = int(nframes)
            self.fps[name] = float(fps)
        
        self.files = open('splits/{}.csv'.format(self.split), 'r').read().splitlines()
        # drop <15 fps videos for audio-only training
        if self.modality == 'audio':
            self.files = [name for name in self.files if self.fps[name] >= 15.0]
        
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
            self.labels_va, self.labels_expr, self.labels_au = {}, {}, {}
            fold_map = {'train': 'Training_Set', 'val': 'Validation_Set'}
            for vid_name in self.files:
                lines = open(os.path.join(self.path, 'annotations', 'VA_Set', fold_map[self.split], vid_name + '.txt'), 'r').read().splitlines()
                # T * 2
                self.labels_va[vid_name] = np.loadtxt(lines, delimiter=',', skiprows=1, dtype=np.float32)
            # load expression labels
            for l in open('splits/expr.csv', 'r').read().splitlines():
                vid_name, expr_split = l.split(',')
                lines = open(os.path.join(self.path, 'annotations', 'EXPR_Set', expr_split, vid_name + '.txt'), 'r').read().splitlines()
                # T * 1
                self.labels_expr[vid_name] = np.loadtxt(lines, skiprows=1, dtype=np.int64)
            # load AU labels
            '''
            for l in open('splits/au.csv', 'r').read().splitlines():
                vid_name, au_split = l.split(',')
                lines = open(os.path.join(self.path, 'annotations', 'AU_Set', au_split, vid_name + '.txt'), 'r').read().splitlines()
                # T * 8
                self.labels_au[vid_name] = np.loadtxt(lines, delimiter=',', skiprows=1, dtype=np.int64)
            '''
        if self.split == 'train':
            if noise_and_balance:
                self.avail_windows = self.get_noisy_balanced_windows()
            else:
                self.avail_windows = self.get_available_windows()
        
        print ('Loaded partition {}: {} files, {} windows'.format(self.split, len(self.files), len(self.sample_src)))
    
    def get_noisy_balanced_windows(self):
        windows = {k: [] for k in self.files}
        cache_path = '{}_{}_noisybalancedwindow{}_{}.pkl'.format(self.release, self.split, self.window_len, self.modality)
        if os.path.exists(cache_path):
            return pickle.load(open(cache_path, 'rb'))
        if self.modality == 'audio':
            # audio only
            for vid_name in tqdm(self.files, desc='Scanning available windows'):
                has_label = np.max(np.abs(self.labels_va[vid_name]), axis=1) <= 1
                avail_ranges = one_runs(has_label)
                for w_st, w_ed in avail_ranges:
                    windows[vid_name].extend(list(range(w_st, w_ed - self.window_len + 1)))
                # try to balance the dataset by considering valence
                scores = [(window, self.labels_va[vid_name][w_st: w_st + self.window_len, 0].mean()) for window, (w_st, _) in zip(windows[vid_name], avail_ranges)]
                scores.sort(key=lambda x: x[1])
                zero_crossing = np.searchsorted([x[1] for x in scores], 0)
                # duplicate all windows with negative mean valence
                windows[vid_name].extend([x[0] for x in scores[: zero_crossing]])
        else:
            # visual and audiovisual
            for vid_name in tqdm(self.files, desc='Scanning available windows'):
                src_fold = os.path.join(self.base, vid_name)
                has_image = np.array([os.path.exists(os.path.join(src_fold, '{:05d}.jpg'.format(i + 1))) for i in range(len(self.labels_va[vid_name]))])
                has_label = np.max(np.abs(self.labels_va[vid_name]), axis=1) <= 1
                # force noisy unlabelled frames to zero
                self.labels_va[vid_name][~has_label] = 0
                # TODO(yuanhang): this sliding process is probably slow but should work
                for w_st in range(0, self.nb_frames[vid_name] - self.window_len + 1):
                    # allow up to 25% missing frames or missing labels
                    missing_percentage = max(1 - np.sum(has_image[w_st: w_st + self.window_len]) / self.window_len, 1 - np.sum(has_label[w_st: w_st + self.window_len]) / self.window_len)
                    if missing_percentage > 0.25: continue
                    windows[vid_name].append(w_st)
                    # duplicate windows with negative mean valence by double appending
                    if self.labels_va[vid_name][w_st: w_st + self.window_len, 0].mean() < 0:
                        windows[vid_name].append(w_st)
                # we're up all night to get lucky :(
                assert len(windows[vid_name]) > 0, 'no available windows for {}'.format(vid_name)
        pickle.dump(windows, open(cache_path, 'wb'))
        return windows
    
    def get_available_windows(self):
        windows = {k: [] for k in self.files}
        cache_path = '{}_{}_window{}_{}.pkl'.format(self.release, self.split, self.window_len, self.modality)
        if os.path.exists(cache_path):
            return pickle.load(open(cache_path, 'rb'))
        if self.modality == 'audio':
            for vid_name in tqdm(self.files, desc='Scanning available windows'):
                has_label = np.max(np.abs(self.labels_va[vid_name]), axis=1) <= 1
                avail_ranges = one_runs(has_label)
                for w_st, w_ed in avail_ranges:
                    windows[vid_name].extend(list(range(w_st, w_ed - self.window_len + 1)))
        else:
            for vid_name in tqdm(self.files, desc='Scanning available windows'):
                src_fold = os.path.join(self.base, vid_name)
                # check that both image and annotation is available
                has_image = np.array([os.path.exists(os.path.join(src_fold, '{:05d}.jpg'.format(i + 1))) for i in range(len(self.labels_va[vid_name]))])
                has_label = np.max(np.abs(self.labels_va[vid_name]), axis=1) <= 1
                avail_ranges = one_runs(has_image & has_label)
                for w_st, w_ed in avail_ranges:
                    windows[vid_name].extend(list(range(w_st, w_ed - self.window_len + 1)))
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
        
        # note that frame indices in filenames begin with 1
        if 'visual' in self.modality:
            is_training = self.split == 'train'
            src_vid_fold = os.path.join(self.base, vid_name)
            inputs = load_video(src_vid_fold, start_frame, track_len,
                                is_training,
                                random.random() > 0.5,
                                self.release == 'vipl',
                                self.apply_cutout,
                                self.input_size)
            se_path = os.path.join(self.path, 'se101_feats', vid_name + '.npy')
            se_features = np.load(se_path)[start_frame: start_frame + track_len].transpose() # 512 * T
            au_path = os.path.join(self.path, 'AU_feats', vid_name + '.npy')
            # TODO(yuanhang): we drop the 12 AU scores for now
            au_features = np.load(au_path)[start_frame: start_frame + track_len, :256].transpose() # 256 * T
        if 'audio' in self.modality:
            if self.fps[vid_name] < 15:
                audio = np.zeros((self.window_len, 200), dtype=np.float32)
            else:
                src_aud_fold = os.path.join(self.path, 'mel_spec', vid_name + '.npy')
                audio = load_audio(src_aud_fold, start_frame, track_len)
        
        if self.split != 'test':
            va_labels = self.labels_va[vid_name][start_frame: start_frame + track_len]
            has_expr = vid_name in self.labels_expr.keys()
            if has_expr:
                expr_labels = self.labels_expr[vid_name][start_frame: start_frame + track_len]
            else:
                expr_labels = np.zeros(track_len, dtype=np.int64)
            expr_valid = np.array([has_expr] * track_len) & (expr_labels >= 0)
            expr_labels = np.clip(expr_labels, 0, 6) # prune invalid labels: 0 <= class_id <= 6
            '''
            has_au = vid_name in self.labels_au.keys()
            if has_au:
                au_labels = self.labels_au[vid_name][start_frame: start_frame + track_len]
            else:
                au_labels = np.zeros((track_len, 8), dtype=np.int64)
            au_valid = np.array([has_au] * track_len) & (np.min(au_labels) >= 0)
            au_labels = np.clip(au_labels, 0, 1)
            '''
        # pad with boundary values, which will be discarded for evaluation
        to_pad = self.window_len - track_len
        if to_pad != 0:
            if 'visual' in self.modality:
                inputs = np.pad(inputs, ((0, 0), (0, to_pad), (0, 0), (0, 0)), 'edge') # (C, T, H, W)
                se_features = np.pad(se_features, ((0, 0), (0, to_pad)), 'edge') # (C, T)
                au_features = np.pad(au_features, ((0, 0), (0, to_pad)), 'edge') # (C, T)
            if 'audio' in self.modality:
                audio = np.pad(audio, ((0, to_pad), (0, 0)), 'edge') # (T, C)
            if self.split != 'test':
                va_labels = np.pad(va_labels, ((0, to_pad), (0, 0)), 'edge')
                expr_labels = np.pad(expr_labels, ((0, to_pad)), 'edge')
                # au_labels = np.pad(au_labels, ((0, to_pad), (0, 0)), 'edge')
                expr_valid = np.pad(expr_valid, ((0, to_pad)), 'edge')
                au_valid = np.pad(au_valid, ((0, to_pad)), 'edge')

        batch = {
            'vid_name': vid_name,
            'start': start_frame,
            'length': track_len
        }
        # add data
        if 'visual' in self.modality:
            batch['video'] = torch.from_numpy(inputs)
            batch['se_features'] = torch.from_numpy(se_features)
            batch['au_features'] = torch.from_numpy(au_features)
        if 'audio' in self.modality:
            batch['audio'] = torch.from_numpy(audio)
        # add labels
        if self.split != 'test':
            batch['label_valence'] = torch.from_numpy(va_labels[..., 0])
            batch['class_expr'] = expr_labels
            batch['expr_valid'] = expr_valid
            # batch['class_au'] = au_labels
            # batch['au_valid'] = au_valid
            batch['label_arousal'] = torch.from_numpy(va_labels[..., 1])
        
        return batch

    def __len__(self):
        return len(self.sample_src)
