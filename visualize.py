from models.utils import smooth_predictions, plot_results

from matplotlib import pyplot as plt

import torch
import numpy as numpy
import sys

BASE_IMG_DIR = '/media/yuan-hang/SSD_DATA/Aff-Wild2/face_128'
BASE_AUD_DIR = '/media/yuan-hang/SSD_DATA/Aff-Wild2/audio'


if __name__ == '__main__':
    data = torch.load(sys.argv[1])
    video_name = input('Video name: ')
    print ('{} frames'.format(len(data['valence_gt'][video_name])))
    start = int(input('Start frame: '))
    end = int(input('End frame: '))
    plot_results(BASE_IMG_DIR, data['valence_gt'][video_name][start: end],
                data['valence_pred'][video_name][start: end], 'valence')  
    plot_results(BASE_AUD_DIR, data['arousal_gt'][video_name][start: end],
                data['arousal_pred'][video_name][start: end], 'arousal')
