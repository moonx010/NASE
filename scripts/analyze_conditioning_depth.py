#!/usr/bin/env python
"""
Analyze conditioning injection depth: measure layer-wise activation
difference between conditioned and unconditioned forward passes.

For the deep injection (temb) model:
- Conditioned: extra_cond = encoder output (normal)
- Unconditioned: extra_cond = zeros

This shows that deep injection maintains conditioning signal across ALL
~37 ResBlocks, while input addition (by design) only affects layer 0.

Usage:
    cd /home/nas4_user/kyudanjung/seokhoonmoon/NASE
    conda activate sgmse
    CUDA_VISIBLE_DEVICES=3 python scripts/analyze_conditioning_depth.py \
        --ckpt logs/multi-deg-wavlm-v2/epoch=159-last.ckpt \
        --pretrain_class_model /home/nas4_user/kyudanjung/seokhoonmoon/pretrained_models/WavLM-Large.pt \
        --test_dir /home/nas4_user/kyudanjung/seokhoonmoon/data/multi_degradation_16k/test_multi \
        --output_dir analysis/conditioning_depth \
        --n_samples 50
"""

import argparse
import json
import os
import sys
import glob

import torch
import numpy as np
from torchaudio import load

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgmse.model import ScoreModel
from sgmse.util.other import pad_spec


def register_hooks(model_dnn, storage_dict):
    """Register forward hooks on all ResBlock-like modules in NCSN++.

    ResNetBlock_* modules take (h, temb) as input, so we identify them
    by checking if they have a Dense layer for temb projection.
    """
    hooks = []
    layer_idx = 0

    for i, module in enumerate(model_dnn.all_modules):
        module_name = type(module).__name__
        # ResNetBlock modules are the ones that receive temb
        if 'ResnetBlock' in module_name:
            name = f"layer_{layer_idx:02d}_{module_name}"
            storage_dict[name] = []

            def make_hook(n):
                def hook_fn(mod, inp, out):
                    storage_dict[n].append(out.detach())
                return hook_fn

            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
            layer_idx += 1

    return hooks


def forward_conditioned(model, Y, y_wav, t):
    """Run conditioned forward pass (normal inference)."""
    x = Y  # Use noisy as starting point
    dnn_input = torch.cat([x, Y], dim=1)

    # Get conditioning
    enc_out = model.noise_encoder(y_wav.squeeze(1))
    noise_emb = enc_out[0].transpose(1, 2).contiguous()
    shared = model.post_cnn(noise_emb).mean(dim=-1)

    noise_feat = model.noise_proj(shared)
    reverb_feat = model.reverb_proj(shared)
    distort_feat = model.distort_proj(shared)

    combined = torch.cat([noise_feat, reverb_feat, distort_feat], dim=-1)
    extra_cond = model.cond_to_temb(combined)

    score = -model.dnn(dnn_input, t, extra_cond=extra_cond)
    return score, extra_cond


def forward_unconditioned(model, Y, t):
    """Run unconditioned forward pass (zero conditioning)."""
    x = Y
    dnn_input = torch.cat([x, Y], dim=1)
    extra_cond = torch.zeros(Y.shape[0], 512, device=Y.device)
    score = -model.dnn(dnn_input, t, extra_cond=extra_cond)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--pretrain_class_model", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="analysis/conditioning_depth")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--inject_method", type=str, default="temb", choices=("temb", "addition"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.ckpt}")
    model = ScoreModel.load_from_checkpoint(
        args.ckpt,
        base_dir='',
        batch_size=16,
        num_workers=0,
        pretrain_class_model=args.pretrain_class_model,
        encoder_type="wavlm",
        multi_degradation=True,
        inject_method=args.inject_method,
        kwargs=dict(gpu=False),
    )
    model.eval(no_ema=False)
    model.cuda()

    # Get test files
    noisy_dir = os.path.join(args.test_dir, "noisy")
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))[:args.n_samples]
    print(f"Using {len(noisy_files)} samples")

    # Collect results across samples
    all_relative_diffs = {}
    all_cosine_sims = {}

    # Use a single timestep for analysis (mid-range)
    t_value = 0.5  # mid-point of diffusion process

    for file_idx, noisy_file in enumerate(noisy_files):
        if file_idx % 10 == 0:
            print(f"Processing {file_idx+1}/{len(noisy_files)}")

        # Load and prepare
        y, sr = load(noisy_file)
        y_wav = y.clone()
        norm_factor = y.abs().max()
        y = y / norm_factor

        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        y_wav = y_wav.unsqueeze(0).cuda() if y_wav.dim() == 1 else y_wav.cuda()

        t = torch.tensor([t_value], device='cuda')

        with torch.no_grad():
            # --- Conditioned pass ---
            cond_activations = {}
            hooks_cond = register_hooks(model.dnn, cond_activations)

            score_cond, extra_cond = forward_conditioned(model, Y, y_wav, t)

            for h in hooks_cond:
                h.remove()

            # --- Unconditioned pass ---
            uncond_activations = {}
            hooks_uncond = register_hooks(model.dnn, uncond_activations)

            score_uncond = forward_unconditioned(model, Y, t)

            for h in hooks_uncond:
                h.remove()

            # --- Compute differences ---
            for name in cond_activations:
                if name not in uncond_activations:
                    continue
                act_c = cond_activations[name][0].float()
                act_u = uncond_activations[name][0].float()

                # Relative L2 difference: ||cond - uncond|| / ||uncond||
                diff_norm = (act_c - act_u).norm().item()
                uncond_norm = act_u.norm().item()
                rel_diff = diff_norm / (uncond_norm + 1e-8)

                # Cosine similarity (flatten)
                cos_sim = torch.nn.functional.cosine_similarity(
                    act_c.flatten().unsqueeze(0),
                    act_u.flatten().unsqueeze(0)
                ).item()

                if name not in all_relative_diffs:
                    all_relative_diffs[name] = []
                    all_cosine_sims[name] = []
                all_relative_diffs[name].append(rel_diff)
                all_cosine_sims[name].append(cos_sim)

    # --- Aggregate ---
    print("\n" + "=" * 80)
    print(f"Layer-wise Conditioning Effect (inject_method={args.inject_method})")
    print(f"{'Layer':<45} {'RelDiff (mean±std)':<25} {'CosSim (mean±std)':<25}")
    print("-" * 95)

    results = {}
    for name in sorted(all_relative_diffs.keys()):
        rd = np.array(all_relative_diffs[name])
        cs = np.array(all_cosine_sims[name])
        results[name] = {
            "rel_diff_mean": float(rd.mean()),
            "rel_diff_std": float(rd.std()),
            "cosine_sim_mean": float(cs.mean()),
            "cosine_sim_std": float(cs.std()),
        }
        print(f"{name:<45} {rd.mean():.4f} ± {rd.std():.4f}      {cs.mean():.6f} ± {cs.std():.6f}")

    # Extra conditioning stats
    print(f"\n{'Extra cond norm':<45} {extra_cond.norm().item():.4f}")

    # Score difference
    score_diff = (score_cond - score_uncond).norm().item()
    score_norm = score_uncond.norm().item()
    print(f"{'Score relative diff':<45} {score_diff / (score_norm + 1e-8):.4f}")

    # Save
    output = {
        "inject_method": args.inject_method,
        "n_samples": len(noisy_files),
        "t_value": t_value,
        "layers": results,
        "extra_cond_norm": extra_cond.norm().item(),
        "score_rel_diff": score_diff / (score_norm + 1e-8),
    }

    json_path = os.path.join(args.output_dir, f"depth_analysis_{args.inject_method}.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Also analyze at multiple timesteps
    print("\n\n===== Multi-timestep analysis =====")
    timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Use first 10 samples for speed
    quick_files = noisy_files[:10]
    timestep_results = {}

    for t_val in timesteps:
        t = torch.tensor([t_val], device='cuda')
        layer_diffs = {}

        for noisy_file in quick_files:
            y, sr = load(noisy_file)
            y_wav = y.clone()
            norm_factor = y.abs().max()
            y = y / norm_factor
            Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
            Y = pad_spec(Y)
            y_wav = y_wav.unsqueeze(0).cuda() if y_wav.dim() == 1 else y_wav.cuda()

            with torch.no_grad():
                cond_act = {}
                hooks_c = register_hooks(model.dnn, cond_act)
                forward_conditioned(model, Y, y_wav, t)
                for h in hooks_c: h.remove()

                uncond_act = {}
                hooks_u = register_hooks(model.dnn, uncond_act)
                forward_unconditioned(model, Y, t)
                for h in hooks_u: h.remove()

                for name in cond_act:
                    if name not in uncond_act:
                        continue
                    diff = (cond_act[name][0].float() - uncond_act[name][0].float()).norm().item()
                    base = uncond_act[name][0].float().norm().item()
                    rd = diff / (base + 1e-8)
                    if name not in layer_diffs:
                        layer_diffs[name] = []
                    layer_diffs[name].append(rd)

        timestep_results[str(t_val)] = {
            name: float(np.mean(vals)) for name, vals in layer_diffs.items()
        }

        avg_diff = np.mean([np.mean(v) for v in layer_diffs.values()])
        print(f"t={t_val:.1f}: avg relative diff = {avg_diff:.4f}")

    output["timestep_analysis"] = timestep_results
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to {json_path}")


if __name__ == "__main__":
    main()
