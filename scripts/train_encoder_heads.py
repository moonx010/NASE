"""
Stage 1: Train encoder pipeline (WavLM → post_cnn → projs → heads).

Trains the degradation classification/regression heads so that
proj outputs contain well-separated degradation information.
After this, freeze encoder pipeline and train score network (Stage 2).

Usage:
    python scripts/train_encoder_heads.py \
        --base_dir /path/to/multi_degradation_16k \
        --epochs 30 --batch_size 32 --lr 1e-3 --gpus 1
"""

import argparse
import csv
import os
from glob import glob
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Noise type mapping (same as data_module.py)
noise2id = {
    'babble': 0, 'cafeteria': 1, 'car': 2, 'kitchen': 3, 'meeting': 4,
    'metro': 5, 'restaurant': 6, 'ssn': 7, 'station': 8, 'traffic': 9,
    'none': 10
}


class MultiDegWavDataset(Dataset):
    """Simple waveform dataset with multi-degradation labels."""

    def __init__(self, data_dir, subset, max_len_sec=4.0, sr=16000):
        self.noisy_files = sorted(glob(join(data_dir, subset, 'noisy', '*.wav')))
        self.max_samples = int(max_len_sec * sr)

        csv_path = join(data_dir, subset, 'labels.csv')
        self.labels = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.labels[row['filename']] = row

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, i):
        wav, sr = torchaudio.load(self.noisy_files[i])
        wav = wav[0]  # mono (T,)

        # Truncate or pad to fixed length
        if wav.shape[0] > self.max_samples:
            start = np.random.randint(0, wav.shape[0] - self.max_samples)
            wav = wav[start:start + self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = F.pad(wav, (0, self.max_samples - wav.shape[0]))

        filename = self.noisy_files[i].split('/')[-1]
        label_row = self.labels.get(filename, None)
        if label_row is not None:
            noise_label = noise2id.get(label_row['noise_type'], 10)
            reverb_label = float(label_row['reverb_t60'])
            distort_label = float(label_row['distort_intensity'])
        else:
            noise_label, reverb_label, distort_label = 10, 0.0, 0.0

        return wav, noise_label, reverb_label, distort_label


class EncoderHeadsModel(pl.LightningModule):
    """WavLM (frozen) → post_cnn → 3 projs → 3 heads."""

    def __init__(self, lr=1e-3, freeze_wavlm=True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # WavLM Base (frozen)
        bundle = torchaudio.pipelines.WAVLM_BASE
        self.wavlm = bundle.get_model()
        if freeze_wavlm:
            self.wavlm.eval()
            for p in self.wavlm.parameters():
                p.requires_grad = False

        # post_cnn: 768 → 256 (same architecture as ScoreModel)
        self.post_cnn = nn.Sequential(
            nn.Conv1d(768, 256, 1, 1, 0),
            nn.PReLU(),
            nn.Conv1d(256, 256, 3, 1, 1)
        )

        # 3-branch projections: 256 → 128 each
        self.noise_proj = nn.Linear(256, 128)
        self.reverb_proj = nn.Linear(256, 128)
        self.distort_proj = nn.Linear(256, 128)

        # Classification / regression heads on 128-dim proj outputs
        self.noise_head = nn.Linear(128, 11)
        self.reverb_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.distort_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # Losses
        self.noise_ce = nn.CrossEntropyLoss()
        self.reverb_mse = nn.MSELoss()
        self.distort_mse = nn.MSELoss()

    def forward(self, wav):
        """wav: (B, T_samples) → proj features + predictions"""
        with torch.no_grad() if self.hparams.freeze_wavlm else torch.enable_grad():
            features, _ = self.wavlm.extract_features(wav)
            emb = features[-1]  # (B, T_frames, 768)

        emb = emb.transpose(1, 2).contiguous()  # (B, 768, T_frames)
        shared = self.post_cnn(emb).mean(dim=-1)  # (B, 256)

        noise_feat = self.noise_proj(shared)      # (B, 128)
        reverb_feat = self.reverb_proj(shared)    # (B, 128)
        distort_feat = self.distort_proj(shared)  # (B, 128)

        noise_logits = self.noise_head(noise_feat)              # (B, 11) raw logits
        reverb_pred = self.reverb_head(reverb_feat).squeeze(-1)  # (B,)
        distort_pred = self.distort_head(distort_feat).squeeze(-1)  # (B,)

        return noise_logits, reverb_pred, distort_pred

    def training_step(self, batch, batch_idx):
        wav, noise_label, reverb_label, distort_label = batch

        noise_logits, reverb_pred, distort_pred = self(wav)

        loss_noise = self.noise_ce(noise_logits, noise_label)
        loss_reverb = self.reverb_mse(reverb_pred, reverb_label.float())
        loss_distort = self.distort_mse(distort_pred, distort_label.float())
        loss = loss_noise + loss_reverb + loss_distort

        # Accuracy
        pred = noise_logits.argmax(dim=1)
        acc = (pred == noise_label).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_noise_loss', loss_noise, on_step=False, on_epoch=True)
        self.log('train_reverb_loss', loss_reverb, on_step=False, on_epoch=True)
        self.log('train_distort_loss', loss_distort, on_step=False, on_epoch=True)
        self.log('train_nc_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wav, noise_label, reverb_label, distort_label = batch

        noise_logits, reverb_pred, distort_pred = self(wav)

        loss_noise = self.noise_ce(noise_logits, noise_label)
        loss_reverb = self.reverb_mse(reverb_pred, reverb_label.float())
        loss_distort = self.distort_mse(distort_pred, distort_label.float())
        loss = loss_noise + loss_reverb + loss_distort

        pred = noise_logits.argmax(dim=1)
        acc = (pred == noise_label).float().mean()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_nc_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_noise_loss', loss_noise, on_epoch=True, sync_dist=True)
        self.log('val_reverb_loss', loss_reverb, on_epoch=True, sync_dist=True)
        self.log('val_distort_loss', loss_distort, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Only optimize trainable params (post_cnn, projs, heads)
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(params, lr=self.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--wandb_name', type=str, default='stage1-encoder-heads')
    parser.add_argument('--save_dir', type=str, default='logs/stage1')
    args = parser.parse_args()

    # Data
    train_ds = MultiDegWavDataset(args.base_dir, 'train')
    val_ds = MultiDegWavDataset(args.base_dir, 'valid')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = EncoderHeadsModel(lr=args.lr, freeze_wavlm=True)

    # Logger & callbacks
    logger = WandbLogger(project='nase-adaptive-guidance', name=args.wandb_name,
                         save_dir='logs')
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.save_dir, filename='best-{epoch}-{val_nc_acc:.3f}',
        monitor='val_nc_acc', mode='max', save_top_k=1, save_last=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=args.gpus,
        logger=logger,
        callbacks=[checkpoint_cb],
        precision='16-mixed',  # WavLM is large, use mixed precision
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_nc_acc: {checkpoint_cb.best_model_score:.4f}")


if __name__ == '__main__':
    main()
