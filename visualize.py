from models.utils import smooth_predictions, plot_results
from matplotlib import pyplot as plt

import torch
import numpy as numpy
import sys


if __name__ == '__main__':
    data = torch.load(sys.argv[1])
    video_name = input('Video name: ')
    print ('{} frames'.format(len(data['valence_gt'][video_name])))
    start = int(input('Start frame: '))
    end = int(input('End frame: '))
    plot_results(data['valence_gt'][video_name][start: end],
                smooth_predictions(data['valence_pred'][video_name][start: end]), 'valence')  
    plot_results(data['arousal_gt'][video_name][start: end],
                smooth_predictions(data['arousal_pred'][video_name][start: end]), 'arousal')
