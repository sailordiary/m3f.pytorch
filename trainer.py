import logging
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from models.model import AffWild2VA

logging.basicConfig(level=logging.INFO)


def main(hparams):
    # init module
    model = AffWild2VA(hparams)

    trainer = Trainer(
        nb_sanity_val_steps=0, # skip sanity check
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
    parser.add_argument('--gpus', type=str, default='2,3')
    parser.add_argument('--nodes', type=int, default=1)

    # give the module a chance to add own params
    parser = AffWild2VA.add_model_specific_args(parser)
    # parse params
    hparams = parser.parse_args()

    main(hparams)
