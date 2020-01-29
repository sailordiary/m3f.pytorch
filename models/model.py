from .dataset import AffWild2iBugSequenceDataset
from .backbone import *
from .utils import *

from argparse import ArgumentParser

from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class AffWild2VA(pl.LightningModule):
    
    def __init__(self, hparams):
        super(AffWild2VA, self).__init__()
        self.hparams = hparams
        if self.hparams.backbone == 'resnet':
            self.net = VA_3DResNet(resnet_ver='v1')
        elif self.hparams.backbone == 'v2p':
            self.net = VA_3DVGGM()
        elif self.hparams.backbone == 'densenet':
            self.net = VA_3DDenseNet()
        
    def forward(self, x):
        # normalize to [-1, 1]
        x = (x - 127.5) / 127.5
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x = batch['video']
        valence, arousal = batch['label_valence'], batch['label_arousal']
        y_hat = self.forward(x)
        valence_hat, arousal_hat = y_hat[..., 0], y_hat[..., 1]
        loss_v = F.mse_loss(valence, valence_hat)
        loss_a = F.mse_loss(arousal, arousal_hat)
        # TODO add CCC loss?
        loss = loss_v + loss_a
        return {'loss': loss,
                'progress_bar': {'loss_v': loss_v, 'loss_a': loss_a, 'loss': loss},
                'log': {'loss_v': loss_v, 'loss_a': loss_a, 'loss': loss}
        }
    
    def validation_step(self, batch, batch_idx):
        v = np.array([], dtype=np.float32)
        a = np.array([], dtype=np.float32)
        v_hat = np.array([], dtype=np.float32)
        a_hat = np.array([], dtype=np.float32)
        
        x = batch['video']
        y_hat = self.forward(x)
        valence_hat, arousal_hat = y_hat[..., 0], y_hat[..., 1]
        lens = batch['length']
        
        if 'label_valence' in batch.keys():
            valence, arousal = batch['label_valence'], batch['label_arousal']
            loss_v = F.mse_loss(valence, valence_hat)
            loss_a = F.mse_loss(arousal, arousal_hat)
            loss = loss_v + loss_a
            # TODO: refactor this mess
            for i, (v_batch, a_batch, vhat_batch, ahat_batch) in enumerate(zip(valence, arousal, valence_hat, arousal_hat)):
                v = np.concatenate((v, v_batch.cpu().numpy()[: lens[i]]))
                a = np.concatenate((a, a_batch.cpu().numpy()[: lens[i]]))
                v_hat = np.concatenate((v_hat, vhat_batch.cpu().numpy()[: lens[i]]))
                a_hat = np.concatenate((a_hat, ahat_batch.cpu().numpy()[: lens[i]]))
            ccc_valence = concordance_cc2(v, v_hat)
            ccc_arousal = concordance_cc2(a, a_hat)
            return {'val_loss_v': loss_v, 'val_loss_a': loss_a, 'val_loss': loss, 'val_ccc_v': ccc_valence, 'val_ccc_a': ccc_arousal}
        else:
            y_hat = self.forward(batch['video'])
            return {'predictions': y_hat,
                    'lengths': batch['length'],
                    'vid_names': batch['vid_name'],
                    'start_frames': batch['start_frame']}
    
    def validation_end(self, outputs):
        if 'val_loss' in outputs[0].keys():
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            avg_loss_v = torch.stack([x['val_loss_v'] for x in outputs]).mean()
            avg_loss_a = torch.stack([x['val_loss_a'] for x in outputs]).mean()
            avg_ccc_v = np.stack([x['val_ccc_v'] for x in outputs]).mean()
            avg_ccc_a = np.stack([x['val_ccc_a'] for x in outputs]).mean()
            return {'progress_bar': {'val_loss': avg_loss, 'val_ccc_v': avg_ccc_v, 'val_ccc_a': avg_ccc_a},
                   'log': {'val_loss': avg_loss, 'val_loss_v': avg_loss_v, 'val_loss_a': avg_loss_a, 'val_ccc_v': avg_ccc_v, 'val_ccc_a': avg_ccc_a}}
        else:
            # TODO: implement test result collection scheme
            # and collect CCC / MSE values
            return outputs

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.hparams.learning_rate,
                                         weight_decay=1e-4)
            return optimizer
#             scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
#             return [optimizer], [scheduler]
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.learning_rate,
                                        momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)
            return [optimizer], [scheduler]
    

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.mode == 'video':
            dataset = AffWild2iBugSequenceDataset('train', self.hparams.dataset_path, self.hparams.window, self.hparams.windows_per_epoch)
        else:
            # TODO: implement framewise
            raise NotImplementedError
        if self.hparams.distributed:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True, sampler=dist_sampler)
        else:
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.workers, pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.mode == 'video':
            dataset = AffWild2iBugSequenceDataset('val', self.hparams.dataset_path, self.hparams.window)
        else:
            raise NotImplementedError
        if self.hparams.distributed:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True, sampler=dist_sampler)
        else:
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.workers, pin_memory=True)

    @pl.data_loader
    def test_dataloader(self):
        if self.hparams.mode == 'video':
            dataset = AffWild2iBugSequenceDataset('test', self.hparams.dataset_path, self.hparams.window)
        else:
            raise NotImplementedError
        if self.hparams.distributed:
            dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.workers, pin_memory=True, sampler=dist_sampler)
        else:
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.workers, pin_memory=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone', default='resnet', type=str)
        parser.add_argument('--mode', default='video', type=str)
        parser.add_argument('--window', default=16, type=int)
        parser.add_argument('--windows_per_epoch', default=200, type=int)
        parser.add_argument('--learning_rate', default=0.0003, type=float)
        parser.add_argument('--batch_size', default=96, type=int)
        parser.add_argument('--optimizer', default='adam', type=str)

        # training specific (for this model)
        parser.add_argument('--distributed', action='store_true', default=False)
        parser.add_argument('--dataset_path', default='/.data/zhangyuanhang/Aff-Wild2', type=str)
        parser.add_argument('--checkpoint_path', default='.', type=str)
        parser.add_argument('--workers', default=8, type=int)
        parser.add_argument('--max_nb_epochs', default=30, type=int)

        return parser
