import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from pytorch_lightning.strategies import DDPStrategy
except ImportError:
    from pytorch_lightning.plugins import DDPPlugin as DDPStrategy

import wandb

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel

import torch


if __name__ == '__main__':
     print('start training')
     parser = ArgumentParser()
     parser.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
     parser.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
     parser.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B")
     parser.add_argument("--wandb_project", type=str, default="nase-adaptive-guidance")
     parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")
     parser.add_argument("--pretrain_class_model", type=str, required=True)
     parser.add_argument("--inject_type", type=str, default="addition")
     parser.add_argument("--max_epochs", type=int, default=160)
     parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")

     # ScoreModel args
     ScoreModel.add_argparse_args(parser)
     # SDE args
     sde_cls = SDERegistry.get_by_name("ouve")
     sde_cls.add_argparse_args(parser)
     # Backbone args
     backbone_cls = BackboneRegistry.get_by_name("ncsnpp")
     backbone_cls.add_argparse_args(parser)
     # DataModule args
     SpecsDataModule.add_argparse_args(parser)

     args = parser.parse_args()

     # Collect kwargs for ScoreModel
     model_kwargs = {}
     for group in parser._action_groups:
          for action in group._group_actions:
               key = action.dest
               val = getattr(args, key, None)
               if key not in ('backbone', 'sde', 'no_wandb', 'wandb_project', 'wandb_name',
                              'pretrain_class_model', 'inject_type', 'max_epochs', 'gpus'):
                    model_kwargs[key] = val

     # Initialize model
     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=SpecsDataModule,
          pretrain_class_model=args.pretrain_class_model,
          inject_type=args.inject_type,
          **model_kwargs
     )

     # Load pre-trained BEATs
     class_checkpoint = torch.load(args.pretrain_class_model)
     model.classfication_model.load_state_dict(class_checkpoint['model'], strict=False)
     for param in model.classfication_model.parameters():
          param.requires_grad = True

     # Set up logger
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir="logs", name="tensorboard")
     else:
          logger = WandbLogger(project=args.wandb_project, name=args.wandb_name, log_model=True, save_dir="logs")
          logger.experiment.log_code(".")

     # Set up callbacks - use wandb_name as dir if available, else logger.version
     log_dir = f"logs/{args.wandb_name}" if args.wandb_name else f"logs/{logger.version}"
     callbacks = [ModelCheckpoint(dirpath=log_dir, save_last=True, filename='{epoch}-last')]
     if args.num_eval_files:
          callbacks.append(ModelCheckpoint(dirpath=log_dir,
               save_top_k=5, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}'))
          callbacks.append(ModelCheckpoint(dirpath=log_dir,
               save_top_k=1, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}'))

     # Initialize Trainer
     trainer = pl.Trainer(
          max_epochs=args.max_epochs,
          accelerator="gpu",
          devices=args.gpus,
          strategy=DDPStrategy(find_unused_parameters=True),
          logger=logger,
          log_every_n_steps=10,
          num_sanity_val_steps=0,
          callbacks=callbacks,
     )

     # Train
     trainer.fit(model)
