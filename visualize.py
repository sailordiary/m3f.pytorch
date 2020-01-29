from models.utils import plot_results

import torch
import numpy as numpy

from matplotlib import pyplot as plt

import sys


if __name__ == '__main__':
    data = torch.load(sys.argv[1])
    video_name = input('Video name:')
    print ('{}: {} frames'.format(len(data['valence_gt'][video_name])))
    plot_results(data['valence_gt'][video_name], data['valence_pred'][video_name], 'valence')  
    plot_results(data['arousal_gt'][video_name], data['arousal_pred'][video_name], 'arousal')
