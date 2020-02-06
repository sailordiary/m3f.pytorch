import os

import math
import random

import numpy as np
import cv2
from .cv_augment import adjust_brightness, adjust_contrast

import torch
from torch.utils.data import Dataset


def load_video(path, start, length,
               is_training=False,
               mirror_augment=False,
               crop_augment=False,
               input_size=128):
    frames = []
    # use identical crop windows for every frame
    if crop_augment:
        if is_training:
            crop_x = random.randint(0, input_size // 8)
            crop_y = random.randint(0, input_size // 8)
        else:
            crop_x, crop_y = input_size // 16, input_size // 16
        crop_size = input_size * 7 // 8
    
    # color jitter
    if is_training:
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)

    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), 'read error: {}'.format(path)
    cap.set(1, start)
    for i in range(length):
        ret, img = cap.read()
        if not ret: img = frames[-1] # TODO(yuanhang): this shouldn't happen
        if crop_augment:
            img = img[crop_y: crop_y + crop_size, crop_x: crop_x + crop_size]
        if mirror_augment and is_training: img = cv2.flip(img, 1)
        if is_training:
            img = adjust_brightness(img, brightness_factor)
            img = adjust_contrast(img, contrast_factor)
        # TODO: add temporal augmentation (repeat, deletion)
        frames.append(img)
    cap.release()
    seq = np.stack(frames).transpose(3, 0, 1, 2).astype(np.float32) # THWC->CTHW
    return seq


class VoxCeleb2Dataset(Dataset):
    '''
    112 by 112 VoxCeleb2 face tracks with 1,000 classes.
    
    Params:
    split: partition (train, val)
    path: base path for data
    window_len: length of temporal crop window
    '''
    def __init__(self, split, path, window_len=16):
        self.split = split
        self.path = path
        self.window_len = window_len

        self.label_map = {l: i for i, l in enumerate(open(os.path.join(self.path, 'vox2_top1000_dev500utt_identity.csv'), 'r').read().splitlines())}
        self.files = []
        for l in open(os.path.join(self.path, 'vox2_top1000_dev500utt_{}.csv'.format(self.split)), 'r').read().splitlines():
            l = l.split('/')
            identity = self.label_map[l[0]]
            path = os.path.join(self.path, 'top1000_64f_128', '{}/{}/{}_{}.mp4'.format(l[0], self.split, l[1], l[2]))
            if os.path.exists(path):
                self.files.append((path, identity))
        
        print ('Loaded partition {}: {} files'.format(self.split, len(self.files)))

    def __getitem__(self, i):
        if self.split == 'train':
            vid_name = self.files[i][0]
            track_len = self.window_len
            start_frame = random.randint(0, 64 - self.window_len)
        else:
            vid_name = self.files[i][0]
            track_len = 24
            start_frame = 16
        
        is_training = self.split == 'train'
        inputs = load_video(vid_name, start_frame, track_len,
                            is_training,
                            random.random() > 0.5,
                            True)
        
        return {
            'video': torch.from_numpy(inputs),
            'label': self.files[i][1]
        }

    def __len__(self):
        return len(self.files)
