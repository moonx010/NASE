import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec
from sgmse.encoders import get_encoder
import logging


class CFGScoreWrapper:
    """Wraps a ScoreModel to apply Classifier-Free Guidance at inference.

    score = (1 + w) * score_cond - w * score_uncond
    where w = guidance_scale.
    """
    def __init__(self, model, guidance_scale):
        self.model = model
        self.guidance_scale = guidance_scale
        # Expose sde for predictor/corrector compatibility
        self.sde = model.sde

    def __call__(self, x, t, y, y_wav):
        score_cond = self.model(x, t, y, y_wav)
        score_uncond = self.model.forward_uncond(x, t, y)
        w = self.guidance_scale
        return (1 + w) * score_cond - w * score_uncond


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--p_uncond", type=float, default=0.0, help="Probability of unconditional training for CFG (0.0 = baseline, no CFG)")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2,
        num_eval_files=20, loss_type='mse', data_module_cls=None,
        pretrain_class_model="/home3/huyuchen/pytorch_workplace/sgmse/BEATs_iter3_plus_AS2M.pt",
        inject_type="addition", p_uncond=0.0, encoder_type="beats", **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
            encoder_type: Noise encoder type ('beats', 'wavlm', 'panns')
        """
        super().__init__()
        if pretrain_class_model is None or len(pretrain_class_model) == 0:
            pretrain_class_model = "/home3/huyuchen/pytorch_workplace/sgmse/BEATs_iter3_plus_AS2M.pt"

        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)

        # Noise encoder (BEATs / WavLM / PANNs)
        self.encoder_type = encoder_type
        encoder_cls = get_encoder(encoder_type)
        self.noise_encoder = encoder_cls(pretrain_class_model=pretrain_class_model)
        encoder_dim = encoder_cls.embed_dim  # 768 for beats/wavlm, 2048 for panns

        # Postprocessing modules — input dim adapts to encoder
        self.post_cnn = torch.nn.Sequential(
            torch.nn.Conv1d(encoder_dim, 256, 1, 1, 0),
            torch.nn.PReLU(),
            torch.nn.Conv1d(256, 256, 3, 1, 1)
        )

        self.classfication_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.inject_type = inject_type
        self.p_uncond = p_uncond
        if self.inject_type == 'concat':
            self.proj = torch.nn.Linear(512, 256)

        # Noise embedding scale for inference (1.0 = normal, 0.0 = no conditioning)
        self.noise_scale = 1.0

        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        # Remap old 'classfication_model.*' keys to 'noise_encoder.model.*' (BEATs compat)
        state = checkpoint.get('state_dict', {})
        remapped = {}
        for k, v in list(state.items()):
            if k.startswith('classfication_model.'):
                new_key = k.replace('classfication_model.', 'noise_encoder.model.', 1)
                remapped[new_key] = state.pop(k)
        state.update(remapped)

        ema = checkpoint.get('ema', None)
        if ema is not None:
            # Also remap EMA shadow params if needed
            if 'shadow_params' in ema:
                # EMA stores params by position, not name — no remapping needed
                pass
            self.ema.load_state_dict(ema)
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step_train(self, batch, batch_idx):
        x, y, y_wav, noise_label = batch
        # print(f'x.shape = {x.shape}, y.shape = {y.shape}')
        # print(f'noise_label = {noise_label}, noise_label.shape = {noise_label.shape}')

        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y, y_wav)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        score, logits = self.forward_train(perturbed_data, t, y, y_wav)
        err = score * sigmas + z
        loss = self._loss(err)

        # classification loss
        loss_class = self.classfication_loss(logits, noise_label)
        loss = loss + loss_class * 0.3

        # classification accuracy
        pred = torch.argmax(logits, dim=1)
        acc = (pred == noise_label).sum() / noise_label.numel()
        # print(f'acc={round(float(acc), 3)}')

        return loss, loss_class, acc

    def training_step(self, batch, batch_idx):
        loss, loss_class, acc = self._step_train(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_nc_loss', loss_class, on_step=True, on_epoch=True)
        self.log('train_nc_acc', acc, on_step=True, on_epoch=True)
        return loss

    def forward_train(self, x, t, y, y_wav):
        # x.shape: (8, 1, 256, 256), y.shape: (8, 1, 256, 256), y_wav.shape: (8, 1, 32640)

        # noise_emb: (B, T_frames, D) -> transpose to (B, D, T_frames)
        noise_emb, logits, _ = self.noise_encoder(y_wav.squeeze(1))
        noise_emb = noise_emb.transpose(1, 2).contiguous()

        # (8, 2, 256, 256) <--> (B, 2, F, T)
        dnn_input = torch.cat([x, y], dim=1)
        T = dnn_input.shape[-1]

        if self.inject_type == 'addition':
            ### 1. addition
            # (8, 1, 256, 1)
            noise_emb = self.post_cnn(noise_emb).mean(dim=-1).unsqueeze(1).unsqueeze(-1)
            # CFG: zero out noise_emb with probability p_uncond
            if self.training and self.p_uncond > 0:
                mask = (torch.rand(x.shape[0], device=x.device) >= self.p_uncond).float()
                noise_emb = noise_emb * mask.view(-1, 1, 1, 1)
            dnn_input = dnn_input + noise_emb
        elif self.inject_type == 'concat':
            ### 2. concat
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            noise_emb = self.post_cnn(noise_emb).mean(dim=-1).unsqueeze(1).unsqueeze(-1).repeat(1, 2, 1, T)
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            mag, phase = torch.abs(dnn_input), torch.angle(dnn_input)
            # (8, 2, 512, 256) <--> (B, 2, 2F, T)
            mag = torch.cat([mag, noise_emb], dim=2)
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            mag = self.proj(mag.transpose(2, 3).contiguous()).transpose(2, 3).contiguous()
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            dnn_input = mag * torch.exp(1j * phase)
        elif self.inject_type == 'cross-attention':
            ### 3. cross-attention
            # (8, 2, 256, 96) <--> (B, 2, F, T')
            noise_emb = self.post_cnn(noise_emb).unsqueeze(1).repeat(1, 2, 1, 1)
            # (8, 2, 96, 256) <--> (B, 2, T', T)
            map = torch.matmul(noise_emb.transpose(2, 3).contiguous(), torch.abs(dnn_input))
            # (8, 2, 1, 256) <--> (B, 2, 1, T)
            map = torch.max(map, dim=2)[0].unsqueeze(2)
            # (8, 2, 256, 1) <--> (B, 2, F, 1)
            beam = (dnn_input * map).sum(dim=-1).unsqueeze(-1)
            dnn_input = dnn_input + beam
        else:
            raise Exception("inject_type not supported")

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score, logits

    # def forward_train(self, x, t, y, y_wav):
    #     # x.shape: (8, 1, 256, 256), y.shape: (8, 1, 256, 256), y_wav.shape: (8, 1, 32640), noise_emb: (8, 96, 768)
    #
    #     # noise_emb: (8, 96, 768)
    #     noise_emb, logits, _ = self.classfication_model(y_wav.squeeze(1))
    #     noise_emb = noise_emb.transpose(1, 2).contiguous()
    #     # (8, 1, 256, 1)
    #     noise_emb = self.post_cnn(noise_emb).mean(dim=-1).unsqueeze(1).unsqueeze(-1)
    #
    #     # Concatenate y as an extra channel
    #     dnn_input = torch.cat([x, y], dim=1) + noise_emb
    #
    #     # the minus is most likely unimportant here - taken from Song's repo
    #     score = -self.dnn(dnn_input, t)
    #     return score, logits

    def _step(self, batch, batch_idx):
        x, y, y_wav = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y, y_wav)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        score = self(perturbed_data, t, y, y_wav)
        err = score * sigmas + z
        loss = self._loss(err)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y, y_wav):
        # x.shape: (8, 1, 256, 256), y.shape: (8, 1, 256, 256), y_wav.shape: (8, 1, 32640)

        # noise_emb: (B, T_frames, D) -> transpose to (B, D, T_frames)
        noise_emb, logits, _ = self.noise_encoder(y_wav.squeeze(1))
        noise_emb = noise_emb.transpose(1, 2).contiguous()

        # (8, 2, 256, 256) <--> (B, 2, F, T)
        dnn_input = torch.cat([x, y], dim=1)
        T = dnn_input.shape[-1]

        if self.inject_type == 'addition':
            ### 1. addition
            # (8, 1, 256, 1)
            noise_emb = self.post_cnn(noise_emb).mean(dim=-1).unsqueeze(1).unsqueeze(-1)
            noise_emb = noise_emb * self.noise_scale  # adaptive scaling
            dnn_input = dnn_input + noise_emb
        elif self.inject_type == 'concat':
            ### 2. concat
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            noise_emb = self.post_cnn(noise_emb).mean(dim=-1).unsqueeze(1).unsqueeze(-1).repeat(1, 2, 1, T)
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            mag, phase = torch.abs(dnn_input), torch.angle(dnn_input)
            # (8, 2, 512, 256) <--> (B, 2, 2F, T)
            mag = torch.cat([mag, noise_emb], dim=2)
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            mag = self.proj(mag.transpose(2, 3).contiguous()).transpose(2, 3).contiguous()
            # (8, 2, 256, 256) <--> (B, 2, F, T)
            dnn_input = mag * torch.exp(1j * phase)
        elif self.inject_type == 'cross-attention':
            ### 3. cross-attention
            # (8, 2, 256, 96) <--> (B, 2, F, T')
            noise_emb = self.post_cnn(noise_emb).unsqueeze(1).repeat(1, 2, 1, 1)
            # (8, 2, 96, 256) <--> (B, 2, T', T)
            map = torch.matmul(noise_emb.transpose(2, 3).contiguous(), torch.abs(dnn_input))
            # (8, 2, 1, 256) <--> (B, 2, 1, T)
            map = torch.max(map, dim=2)[0].unsqueeze(2)
            # (8, 2, 256, 1) <--> (B, 2, F, 1)
            beam = (dnn_input * map).sum(dim=-1).unsqueeze(-1)
            dnn_input = dnn_input + beam
        else:
            raise Exception("inject_type not supported")

        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    # def forward(self, x, t, y, y_wav):
    #     # x.shape: (8, 1, 256, 256), y.shape: (8, 1, 256, 256), y_wav.shape: (8, 1, 32640), noise_emb: (8, 96, 768)
    #
    #     # noise_emb: (8, 96, 768)
    #     noise_emb, logits, _ = self.classfication_model(y_wav.squeeze(1))
    #     noise_emb = noise_emb.transpose(1, 2).contiguous()
    #     # (8, 1, 256, 1)
    #     noise_emb = self.post_cnn(noise_emb).mean(dim=-1).unsqueeze(1).unsqueeze(-1)
    #
    #     # Concatenate y as an extra channel
    #     dnn_input = torch.cat([x, y], dim=1) + noise_emb
    #
    #     # the minus is most likely unimportant here - taken from Song's repo
    #     score = -self.dnn(dnn_input, t)
    #     return score

    def compute_nc_confidence(self, y_wav):
        """Compute NC classifier confidence for adaptive guidance.

        Returns:
            confidence: max softmax probability (scalar or batch)
            logits: raw NC logits (for analysis)
        """
        with torch.no_grad():
            _, logits, _ = self.noise_encoder(y_wav.squeeze(1))
            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1)[0]
        return confidence, logits

    def extract_noise_embedding(self, y_wav, use_raw=False):
        """Extract noise embedding with L2 normalization.

        Args:
            y_wav: (B, 1, T) or (B, T) noisy waveform
            use_raw: if True, use BEATs raw features (768-dim) instead of post_cnn (256-dim)
        Returns:
            embedding: (B, D) L2-normalized embedding
        """
        with torch.no_grad():
            if y_wav.dim() == 3:
                y_wav = y_wav.squeeze(1)
            noise_emb, _, _ = self.noise_encoder(y_wav)       # (B, T_frames, D_enc)
            if use_raw:
                # Use raw encoder features (768-dim for BEATs/WavLM)
                embedding = noise_emb.mean(dim=1)             # (B, D_enc)
            else:
                noise_emb = noise_emb.transpose(1, 2).contiguous()  # (B, D_enc, T_frames)
                noise_emb = self.post_cnn(noise_emb)               # (B, 256, T_frames)
                embedding = noise_emb.mean(dim=-1)                  # (B, 256)
            # L2 normalize — critical for meaningful distances
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def compute_embedding_distance(self, y_wav, ref_embeddings, k=10, tau=1.0, use_raw=False):
        """k-NN distance based adaptive guidance scale.

        Args:
            y_wav: (B, 1, T) noisy waveform
            ref_embeddings: (N_ref, D) L2-normalized reference embeddings
            k: number of nearest neighbors
            tau: temperature for distance-to-weight mapping
        Returns:
            distance: (B,) mean k-NN distance (cosine-like, in [0, 2])
            w: (B,) guidance scale in [-1, 1]
        """
        embedding = self.extract_noise_embedding(y_wav, use_raw=use_raw)
        distances = torch.cdist(embedding, ref_embeddings)       # (B, N_ref)
        knn_dist, _ = torch.topk(distances, k, largest=False, dim=-1)  # (B, k)
        distance = knn_dist.mean(dim=-1)                         # (B,)
        # Map: dist=0 → w=1 (boost cond), dist=τ → w=0, dist→∞ → w=-1 (uncond)
        w = 2.0 / (1.0 + distance / tau) - 1.0
        return distance, w

    def compute_prototype_distance(self, y_wav, prototypes, tau=1.0, use_raw=False):
        """Prototype distance based adaptive guidance scale.

        Args:
            y_wav: (B, 1, T) noisy waveform
            prototypes: (N_classes, D) L2-normalized prototype embeddings
            tau: temperature for distance-to-weight mapping
        Returns:
            distance: (B,) min distance to any prototype
            w: (B,) guidance scale in [-1, 1]
        """
        embedding = self.extract_noise_embedding(y_wav, use_raw=use_raw)
        distances = torch.cdist(embedding, prototypes)           # (B, N_classes)
        distance = distances.min(dim=-1)[0]                      # (B,)
        w = 2.0 / (1.0 + distance / tau) - 1.0
        return distance, w

    def forward_uncond(self, x, t, y):
        """Unconditional score: noise embedding is zeroed out."""
        dnn_input = torch.cat([x, y], dim=1)
        # No noise embedding added (equivalent to zero noise_emb)
        score = -self.dnn(dnn_input, t)
        return score

    def forward_cfg(self, x, t, y, y_wav, guidance_scale=1.0):
        """CFG inference: score = (1+w) * score_cond - w * score_uncond."""
        score_cond = self.forward(x, t, y, y_wav)
        if guidance_scale == 0.0:
            return score_cond
        score_uncond = self.forward_uncond(x, t, y)
        return (1 + guidance_scale) * score_cond - guidance_scale * score_uncond

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, y_wav, N=None, minibatch=None, guidance_scale=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        # Use CFG wrapper if guidance_scale is set
        if guidance_scale is not None and guidance_scale != 0.0:
            score_fn = CFGScoreWrapper(self, guidance_scale)
        else:
            score_fn = self

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=score_fn, y=y, y_wav=y_wav, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    y_wav_mini = y_wav[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=score_fn, y=y_mini, y_wav=y_wav_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1)
        y_wav = y.clone()
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), y_wav.cuda(), N=N,
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
