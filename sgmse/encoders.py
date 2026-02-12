"""
Noise encoder wrappers for NASE with unified interface.

Supported encoders:
- BEATs: Transformer, classification-based SSL, 768-dim (baseline)
- WavLM Base: Transformer, speech SSL (denoising), 768-dim
- PANNs CNN14: CNN, AudioSet classification, 2048-dim
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
from typing import Optional

from sgmse.BEATs import BEATs, BEATsConfig


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

ENCODER_REGISTRY = {}


def register_encoder(name):
    def decorator(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_encoder(name: str) -> type:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODER_REGISTRY.keys())}")
    return ENCODER_REGISTRY[name]


# ──────────────────────────────────────────────
# BEATs Encoder (baseline, existing)
# ──────────────────────────────────────────────

@register_encoder("beats")
class BEATsEncoder(nn.Module):
    """Wraps existing BEATs model. embed_dim=768, predictor_class=10."""

    embed_dim = 768

    def __init__(self, pretrain_class_model: str, predictor_class: int = 10):
        super().__init__()
        checkpoint = torch.load(pretrain_class_model, map_location="cpu")
        cfg = BEATsConfig(checkpoint['cfg'])
        cfg.predictor_class = predictor_class
        self.model = BEATs(cfg)
        # Load pretrained weights
        self.model.load_state_dict(checkpoint['model'], strict=False)

    def forward(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            source: (B, T_samples) raw waveform
        Returns:
            embedding: (B, T_frames, 768)
            logits: (B, predictor_class) — after mean-pool + sigmoid (BEATs native)
            padding_mask
        """
        return self.model(source, padding_mask=padding_mask)


# ──────────────────────────────────────────────
# WavLM Encoder
# ──────────────────────────────────────────────

@register_encoder("wavlm")
class WavLMEncoder(nn.Module):
    """WavLM Base encoder. embed_dim=768, with NC classification head."""

    embed_dim = 768

    def __init__(self, pretrain_class_model: str = None, predictor_class: int = 10):
        super().__init__()
        # Load WavLM Base from torchaudio
        bundle = torchaudio.pipelines.WAVLM_BASE
        self.model = bundle.get_model()
        self.expected_sr = bundle.sample_rate  # 16000

        # NC classification head (same role as BEATs predictor)
        self.predictor_dropout = nn.Dropout(0.1)
        self.predictor = nn.Linear(768, predictor_class)

    def forward(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            source: (B, T_samples) raw waveform at 16kHz
        Returns:
            embedding: (B, T_frames, 768)
            logits: (B, predictor_class) — mean-pooled + sigmoid (matches BEATs interface)
            padding_mask: None
        """
        # WavLM expects (B, T) float waveform
        # extract_features returns list of layer outputs; take last layer
        features, _ = self.model.extract_features(source)
        embedding = features[-1]  # (B, T_frames, 768)

        x = self.predictor_dropout(embedding)
        logits = self.predictor(x)  # (B, T_frames, predictor_class)
        logits = logits.mean(dim=1)  # (B, predictor_class)
        logits = torch.sigmoid(logits)

        return embedding, logits, None


# ──────────────────────────────────────────────
# PANNs CNN14 Encoder
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = torch.relu_(self.bn1(self.conv1(x)))
        x = torch.relu_(self.bn2(self.conv2(x)))
        x = torch.avg_pool2d(x, pool_size)
        return x


class Cnn14(nn.Module):
    """Full CNN14 following PANNs architecture exactly."""

    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)

    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (B, T_frames, 64) log-mel spectrogram
        Returns:
            frame_emb: (B, T_pool, 2048)
            clip_emb: (B, 2048)
        """
        # (B, T, 64) -> (B, 1, T, 64)
        x = mel_spec.unsqueeze(1)

        # BN on mel-freq axis: (B, 1, T, 64) -> (B, 64, T, 1) -> BN -> back
        x = x.transpose(1, 3)  # (B, 64, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)  # (B, 1, T, 64)

        x = self.conv_block1(x, pool_size=(2, 2))  # (B, 64, T/2, 32)
        x = self.conv_block2(x, pool_size=(2, 2))  # (B, 128, T/4, 16)
        x = self.conv_block3(x, pool_size=(2, 2))  # (B, 256, T/8, 8)
        x = self.conv_block4(x, pool_size=(2, 2))  # (B, 512, T/16, 4)
        x = self.conv_block5(x, pool_size=(2, 2))  # (B, 1024, T/32, 2)
        x = self.conv_block6(x, pool_size=(1, 1))  # (B, 2048, T/32, 2)

        # Average over frequency axis → (B, 2048, T/32)
        x = x.mean(dim=3)
        # Transpose to (B, T/32, 2048)
        frame_emb = x.transpose(1, 2)

        # Clip-level: average over time
        clip_emb = frame_emb.mean(dim=1)  # (B, 2048)
        clip_emb = torch.relu_(self.fc1(clip_emb))

        return frame_emb, clip_emb


@register_encoder("panns")
class PANNsEncoder(nn.Module):
    """PANNs CNN14 encoder. embed_dim=2048, with NC classification head."""

    embed_dim = 2048

    def __init__(self, pretrain_class_model: str = None, predictor_class: int = 10):
        super().__init__()
        self.cnn14 = Cnn14()

        # NC classification head
        self.predictor_dropout = nn.Dropout(0.1)
        self.predictor = nn.Linear(2048, predictor_class)

    def _extract_logmel(self, source: torch.Tensor) -> torch.Tensor:
        """Extract 64-bin log-mel spectrogram from waveform, matching PANNs."""
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(
                waveform, num_mel_bins=64, sample_frequency=16000,
                frame_length=25, frame_shift=10
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)  # (B, T_frames, 64)
        return fbank

    def forward(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            source: (B, T_samples) raw waveform at 16kHz
        Returns:
            embedding: (B, T_pool, 2048)
            logits: (B, predictor_class) — sigmoid applied (matches BEATs interface)
            padding_mask: None
        """
        mel = self._extract_logmel(source)  # (B, T_frames, 64)
        frame_emb, clip_emb = self.cnn14(mel)  # frame: (B, T', 2048), clip: (B, 2048)

        # NC head on clip embedding
        x = self.predictor_dropout(clip_emb)
        logits = self.predictor(x)  # (B, predictor_class)
        logits = torch.sigmoid(logits)

        return frame_emb, logits, None

    def load_pretrained_cnn14(self, ckpt_path: str):
        """Load pretrained CNN14 weights from PANNs checkpoint."""
        state = torch.load(ckpt_path, map_location="cpu")
        if "model" in state:
            state = state["model"]

        # Filter to only CNN14 backbone keys
        cnn14_state = {}
        for k, v in state.items():
            # PANNs checkpoint keys: bn0.*, conv_block*.*, fc1.*
            for prefix in ("bn0.", "conv_block", "fc1."):
                if k.startswith(prefix):
                    cnn14_state[k] = v
                    break

        missing, unexpected = self.cnn14.load_state_dict(cnn14_state, strict=False)
        if missing:
            print(f"[PANNs] Missing keys: {missing}")
        if unexpected:
            print(f"[PANNs] Unexpected keys: {unexpected}")
