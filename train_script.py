#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:46:20 2020

@author: eschweiler
"""

from argparse import ArgumentParser

import numpy as np
import torch
import glob
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):

    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    os.makedirs(hparams.output_path, exist_ok=True)
    
    # Load pretrained weights if available
    if not hparams.pretrained is None:
        model.load_pretrained(hparams.pretrained)

    # Resume from checkpoint if available    
    resume_ckpt = None
    if hparams.resume:
        checkpoints = glob.glob(os.path.join(hparams.output_path,'*.ckpt'))
        checkpoints.sort(key=os.path.getmtime)
        if len(checkpoints)>0:
            resume_ckpt = checkpoints[-1]
            print('Resuming from checkpoint: {0}'.format(resume_ckpt))
            
    # Set the augmentations if available
    model.set_augmentations(hparams.augmentations)
        
    # Save a few samples for sanity checks
    print('Saving 20 data samples for sanity checks...')
    model.train_dataloader().dataset.test(os.path.join(hparams.output_path, 'samples'), num_files=20)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.output_path,
        filename=hparams.pipeline+'-{epoch:03d}-{step}',
        save_top_k=1,
        monitor='step',
        mode='max',
        verbose=True,
        every_n_epochs=1
    )
    
    logger = TensorBoardLogger(
        save_dir=hparams.log_path,
        name='lightning_logs_'+hparams.pipeline.lower()
    )
    
    trainer = Trainer(
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        gpus=hparams.gpus,
        min_epochs=hparams.epochs,
        max_epochs=hparams.epochs,
        resume_from_checkpoint=resume_ckpt
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)



if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--output_path',
        type=str,
        default=r'results/experiment1',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--log_path',
        type=str,
        default=r'logs/logs_experiment1',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of GPUs to use'
    )
    
    parent_parser.add_argument(
        '--no_resume',
        dest='resume',
        action='store_false',
        default=True,
        help='Do not resume training from latest checkpoint'
    )
    
    parent_parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        nargs='+',
        help='path to pretrained model weights'
    )
    
    parent_parser.add_argument(
        '--augmentations',
        type=str,
        default=None,
        help='path to augmentation dict file'
    )
    
    parent_parser.add_argument(
        '--epochs',
        type=int,
        default=5000,
        help='number of epochs'
    )
    
    parent_parser.add_argument(
        '--pipeline',
        type=str,
        default='DiffusionModel3D',
        help='which pipeline to load (DiffusionModel3D | DiffusionModel2D)'
    )
    
    parent_args = parent_parser.parse_known_args()[0]
    
    # load the desired network architecture
    if parent_args.pipeline.lower() == 'diffusionmodel3d':
        from models.DiffusionModel3D import DiffusionModel3D as network
    elif parent_args.pipeline.lower() == 'diffusionmodel2d':
        from models.DiffusionModel2D import DiffusionModel2D as network
    else:
        raise ValueError('Pipeline {0} unknown.'.format(parent_args.pipeline))
    
    # each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
