import comet_ml
#import ctypes
#libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import os
import time
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ProgressBar
from pytorch_lightning.loggers import CometLogger

from core.goal_transformer import GoalTransformer

def main(args, params):
    # Fix random seed for reproducibility
    if params['seed'] is not None:
        seed_everything(params['seed'])

    model = GoalTransformer(params)

    # Train
    comet_logger = CometLogger(
        api_key="",
        project_name="goal-transformer",
        experiment_name=params['model_name'],
    )
    checkpoint_callback = ModelCheckpoint(filename=params['model_name'] + "{epoch:03d}", every_n_epochs=1)

    trainer = Trainer(
        gpus=params['num_gpus'] if not args.test else 1, 
        num_nodes=params['num_nodes'] if not args.test else 1, 
        accumulate_grad_batches=params['accumulate_grad_batches'],
        accelerator=params['accelerator'],
        strategy=params['strategy'] if args.train else None,
        precision=32,
#       precision=16,
        gradient_clip_val=params['clip_max_norm'], 
        default_root_dir=params['checkpoint_dir'],
        callbacks=[checkpoint_callback],
        log_every_n_steps=params['log_every_n_steps'],
        logger=comet_logger,
        max_epochs=params['epochs'],
        max_time="07:00:00:00",
        progress_bar_refresh_rate=params['log_every_n_steps'],
    )

    if args.train:
        if params['resume']:
            resume_from = Path(params['checkpoint_dir']) / (params['model_name'] + '_checkpoint.pt')
            trainer.fit(model, ckpt_path=resume_from)
        else:
            trainer.fit(model)
        print('Saving checkpoint...')
        trainer.save_checkpoint(Path(params['checkpoint_dir']) / (params['model_name'] + '_checkpoint.pt'))
        print('Saved checkpoint to ', Path(params['checkpoint_dir']) / (params['model_name'] + '_checkpoint.pt'))

    if args.test:
        resume_from = Path(params['checkpoint_dir']) / (params['model_name'] + '_checkpoint.pt')
        trainer.test(model, ckpt_path=resume_from)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Goal Transformer Training and Evaluation.')
    parser.add_argument('--config', default='config/goal_transformer.yaml')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            params = yaml.safe_load(f)   
        main(args, params)
    else:
        raise ValueError('Configuration file does not exist.')


