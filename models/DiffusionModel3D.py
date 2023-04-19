# -*- coding: utf-8 -*-


import json
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from dataloader.h5_dataloader import MeristemH5Dataset
from ThirdParty.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
 
    
    
class DiffusionModel3D(pl.LightningModule):
    
    def __init__(self, hparams):
        super(DiffusionModel3D, self).__init__()
        
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.augmentation_dict = {}

        # load the backbone network architecture
        if self.hparams.backbone.lower() == 'unet3d_pixelshuffle_inject':
            from models.module_UNet3D_pixelshuffle_inject import module_UNet3D_pixelshuffle_inject as backbone
        else:
            raise ValueError('Unknown backbone architecture {0}!'.format(self.hparams.backbone))
            
        self.network = backbone(patch_size=self.hparams.patch_size, in_channels=self.hparams.in_channels, out_channels=self.hparams.out_channels,\
                                feat_channels=self.hparams.feat_channels, t_channels=self.hparams.t_channels,\
                                out_activation=self.hparams.out_activation, layer_norm=self.hparams.layer_norm)
        # cache for generated images
        self.last_predictions = None
        self.last_imgs = None
        
        # set up diffusion parameters
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.DiffusionTrainer = GaussianDiffusionTrainer(self.hparams.num_timesteps, schedule=self.hparams.diffusion_schedule).to(device)
        self.DiffusionSampler = GaussianDiffusionSampler(self.network, self.hparams.num_timesteps, t_start=self.hparams.num_timesteps,\
                                                         t_save=self.hparams.num_timesteps, t_step=1, schedule=self.hparams.diffusion_schedule,\
                                                         mean_type='epsilon', var_type='fixedlarge').to(device)

    def forward(self, z, t):
        return self.network(z, t)
    
    
    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        
        # Load the state dict
        state_dict = torch.load(pretrained_file)['state_dict']
        
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)
            
        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())
        
        layers = []
        for layer in param_dict:
            if strict and not 'network.'+layer in state_dict:
                if verbose:
                    print('Could not find weights for layer "{0}"'.format(layer))
                continue
            try:
                param_dict[layer].data.copy_(state_dict['network.'+layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print('Error at layer {0}:\n{1}'.format(layer, e))
        
        self.network.load_state_dict(param_dict)
        
        if verbose:
            print('Loaded weights for the following layers:\n{0}'.format(layers))
        
        
    def denoise_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)


    def training_step(self, batch, batch_idx):
        
        # Get image ans mask of current batch
        self.last_imgs = batch['image'].float()
        
        # get x_t, noise for a random t
        self.x_t, noise, t = self.DiffusionTrainer(self.last_imgs[:,0:1,...])
        self.x_t.requires_grad = True
        
        # generate prediction
        self.generated_noise = self.forward(torch.cat((self.x_t, self.last_imgs[:,1:,...]), axis=1), t)
                
        # get the losses
        loss_denoise = self.denoise_loss(self.generated_noise, noise)
        
        self.logger.experiment.add_scalar('loss_denoise', loss_denoise, self.current_epoch) 
        
        return loss_denoise
    
        
    def test_step(self, batch, batch_idx):
        x = batch['image']
        x_hat = self.forward(x, torch.tensor([0,], device=x.device.index))
        return {'test_loss': F.l1_loss(x[:,0:1,...]-x_hat, x[:,0:1,...])} 

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x = batch['image']
        x_hat = self.forward(x, torch.tensor([0,], device=x.device.index))
        return {'val_loss': F.l1_loss(x[:,0:1,...]-x_hat, x[:,0:1,...])} 

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.network.parameters(), lr=self.hparams.learning_rate)
        return [opt], []

    def train_dataloader(self):
         if self.hparams.train_list is None:
            return None
         else:
            dataset = MeristemH5Dataset(self.hparams.train_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, reduce_dim=False,\
                                        augmentation_dict=self.augmentation_dict, samples_per_epoch=self.hparams.samples_per_epoch,\
                                        data_norm=self.hparams.data_norm, no_mask=True, boundary_handling='none', \
                                        image_noise_channel=self.hparams.image_noise_channel, mask_noise_channel=self.hparams.mask_noise_channel, noise_type=self.hparams.noise_type)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
    
    def test_dataloader(self):
        if self.hparams.test_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.test_list, self.hparams.data_root, patch_size=self.hparams.patch_size, reduce_dim=False,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        data_norm=self.hparams.data_norm, no_mask=True, boundary_handling='none',\
                                        image_noise_channel=self.hparams.image_noise_channel, mask_noise_channel=self.hparams.mask_noise_channel, noise_type=self.hparams.noise_type)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        if self.hparams.val_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.val_list, self.hparams.data_root, patch_size=self.hparams.patch_size, reduce_dim=False,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        data_norm=self.hparams.data_norm, no_mask=True, boundary_handling='none',\
                                        image_noise_channel=self.hparams.image_noise_channel, mask_noise_channel=self.hparams.mask_noise_channel, noise_type=self.hparams.noise_type)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)


    def on_train_epoch_end(self):
        
        
        self.DiffusionSampler.model = self.network
        
        with torch.no_grad():
            
            input_patch = self.last_imgs
            
            # get x_0
            x_0,_ = self.DiffusionSampler(input_patch[:,0:1,...], input_patch[:,1:,...])
             
            # log sampled images
            prediction_grid = torchvision.utils.make_grid(x_0[...,int(self.hparams.patch_size[0]//2),:,:])
            self.logger.experiment.add_image('predicted_x_0', prediction_grid, self.current_epoch)
            
            img_grid = torchvision.utils.make_grid(input_patch[...,int(self.hparams.patch_size[0]//2),:,:])
            self.logger.experiment.add_image('raw_x_0', img_grid, self.current_epoch)
            
            
    def set_augmentations(self, augmentation_dict_file):
        if not augmentation_dict_file is None:
            self.augmentation_dict = json.load(open(augmentation_dict_file))
        
        
    @staticmethod
    def add_model_specific_args(parent_parser): 
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--backbone', default='UNet3D_PixelShuffle_inject', type=str, help='which model to load (UNet3D_PixelShuffle_inject)')
        parser.add_argument('--in_channels', default=2, type=int)
        parser.add_argument('--out_channels', default=1, type=int)
        parser.add_argument('--feat_channels', default=16, type=int)
        parser.add_argument('--t_channels', default=128, type=int)
        parser.add_argument('--patch_size', default=(64,128,128), type=int, nargs='+')
        parser.add_argument('--layer_norm', default='instance', type=str)
        parser.add_argument('--out_activation', default='none', type=str)

        # data
        parser.add_argument('--data_norm', default='minmax_shifted', type=str)        
        parser.add_argument('--data_root', default='/data/root', type=str) 
        parser.add_argument('--train_list', default='/path/to/training_data/split1_train.csv', type=str)
        parser.add_argument('--test_list', default='/path/to/testing_data/split1_test.csv', type=str)
        parser.add_argument('--val_list', default='/path/to/validation_data/split1_val.csv', type=str)
        parser.add_argument('--image_groups', default=('data/image',), type=str, nargs='+')
        parser.add_argument('--mask_groups', default=('data/diffusion_mask',), type=str, nargs='+')
        parser.add_argument('--image_noise_channel', default=-1, type=int)
        parser.add_argument('--mask_noise_channel', default=-1, type=int)
        parser.add_argument('--noise_type', default='gaussian', type=str) 
        
        # diffusion parameter
        parser.add_argument('--num_timesteps', default=1000, type=int)
        parser.add_argument('--diffusion_schedule', default='cosine', type=str)
        
        # training params
        parser.add_argument('--samples_per_epoch', default=-1, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        
        return parser