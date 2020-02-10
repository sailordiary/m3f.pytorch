import argparse
import os
import time
import logging

from tqdm import tqdm

import torch
import numpy as np

from models.utils import smooth_predictions


def run_ensemble(eval_list, score_list):
    os.makedirs('VA-Track', exist_ok=True)
    video_names = open(eval_list.name, 'r').read().splitlines()
    video_scores = {k: {'valence': None, 'arousal': None} for k in video_names}
    score_names = open(score_list.name, 'r').read().splitlines()
    nb_scores = len(score_names)
    for i, fname in enumerate(score_names):
        scores = torch.load(fname)
        if i == 0:
            for video_name in video_names:
                video_scores[video_name]['valence'] = scores['valence_pred'][video_name]
                video_scores[video_name]['arousal'] = scores['arousal_pred'][video_name]
        else:
            for video_name in video_names:
                video_scores[video_name]['valence'] += scores['valence_pred'][video_name]
                video_scores[video_name]['arousal'] += scores['arousal_pred'][video_name]
    for video_name in tqdm(video_names):
        with open(os.path.join('VA-Track', video_name + '.txt'), 'w') as fp:
            valence = video_scores[video_name]['valence'].numpy() / nb_scores
            arousal = video_scores[video_name]['arousal'].numpy() / nb_scores
            valence = smooth_predictions(valence)
            arousal = smooth_predictions(arousal)
            fp.write('valence,arousal\n')
            for v, a in zip(valence, arousal):
                fp.write('{:.3f},{:.3f}\n'.format(v, a))
            logging.debug('Done writing %d lines for video %s', len(valence), video_name)


def parse_arguments():
    """Parses command-line flags.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--eval_list",
        help="Text file containing names of videos to be evaluated on.",
        type=argparse.FileType("r"),
        required=True)
    parser.add_argument(
        "-s",
        "--score_list",
        help="Text file containing paths of input prediction .pt files.",
        type=argparse.FileType("r"),
        required=True)
    parser.add_argument(
        "-v", "--verbose", help="Increase output verbosity.", action="store_true")
    return parser.parse_args()
  
  
def main():
    start = time.time()
    args = parse_arguments()
    if args.verbose:
      logging.basicConfig(level=logging.DEBUG)
    del args.verbose
    run_ensemble(**vars(args))
    logging.info("Computed in %s seconds", time.time() - start)


if __name__ == "__main__":
    main()
