import logging
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from models.model import AffWild2VA

import torch
import random
import numpy as np

logging.basicConfig(level=logging.INFO)


def main(hparams):
    torch.backends.cudnn.deterministic = True
    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)

    # init module
    model = AffWild2VA(hparams)

    trainer = Trainer(
        early_stop_callback=None,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
        default_save_path=hparams.checkpoint_path,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        distributed_backend='ddp' if hparams.distributed else 'dp'
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default='2')
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12345)

    # give the module a chance to add own params
    parser = AffWild2VA.add_model_specific_args(parser)
    # parse params
    hparams = parser.parse_args()

    main(hparams)
