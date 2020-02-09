import torch
import logging
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from models.model import AffWild2VA

logging.basicConfig(level=logging.INFO)


def main(hparams):
    # init module
    model = AffWild2VA(hparams)
    # make it easier for us to add new params
    checkpoint = torch.load(hparams.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print ('Loaded pretrained weights')

    trainer = Trainer(
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        distributed_backend='ddp' if hparams.distributed else 'dp'
    )
    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default='2')
    parser.add_argument('--nodes', type=int, default=1)

    parser.add_argument('--checkpoint', type=str, default='')

    # give the module a chance to add own params
    parser = AffWild2VA.add_model_specific_args(parser)
    # parse params
    hparams = parser.parse_args()

    main(hparams)
