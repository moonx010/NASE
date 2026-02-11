import glob
import csv
from argparse import ArgumentParser
from os.path import join

import torch
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--test_set", type=str, default='noisy')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--pretrain_class_model", type=str, required=True)
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--guidance_scale", type=float, default=None, help="CFG guidance scale w. None=no CFG (baseline), 0.0=conditional only, >0=CFG")
    # Adaptive guidance args
    parser.add_argument("--adaptive_guidance", action='store_true', help="Enable adaptive guidance (overrides --guidance_scale)")
    parser.add_argument("--w_max", type=float, default=1.0, help="Max guidance scale for adaptive mode")
    parser.add_argument("--guidance_mapping", type=str, default="linear", choices=("linear", "scaled", "binary"), help="Confidence-to-w mapping: linear=w_max*(2c-1), scaled=w_max*c, binary=w_max if c>0.5 else 0")
    args = parser.parse_args()

    noisy_dir = join(args.test_dir, args.test_set)
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector
    pretrain_class_model = args.pretrain_class_model

    target_dir = args.enhanced_dir
    ensure_dir(target_dir)

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Load score model
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file,
        base_dir='',
        batch_size=16,
        num_workers=0,
        pretrain_class_model=pretrain_class_model,
        kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    if args.adaptive_guidance:
        mode = f"Adaptive CFG (w_max={args.w_max}, mapping={args.guidance_mapping})"
    elif args.guidance_scale is not None:
        mode = "CFG w={:.1f}".format(args.guidance_scale)
    else:
        mode = "baseline (no CFG)"
    print(f'start inference! mode={mode}, N={N}, files={len(noisy_files)}')

    # Per-file guidance log for adaptive mode
    guidance_log = []

    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]

        # Load wav
        y, _ = load(noisy_file)
        T_orig = y.size(1)

        y_wav = y.clone()

        # Normalize
        norm_factor = y.abs().max()
        y = y / norm_factor

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)

        # Determine guidance scale
        if args.adaptive_guidance:
            confidence, logits = model.compute_nc_confidence(y_wav.cuda())
            conf = confidence.item()
            pred_class = torch.argmax(logits, dim=-1).item()
            if args.guidance_mapping == "linear":
                w = args.w_max * (2 * conf - 1)
            elif args.guidance_mapping == "scaled":
                w = args.w_max * conf
            elif args.guidance_mapping == "binary":
                w = args.w_max if conf > 0.5 else 0.0
            guidance_log.append({"filename": filename, "confidence": round(conf, 4), "pred_class": pred_class, "w": round(w, 4)})
            gs = w if abs(w) > 1e-6 else None
        else:
            gs = args.guidance_scale

        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', corrector_cls, Y.cuda(), y_wav.cuda(), N=N,
            corrector_steps=corrector_steps, snr=snr,
            guidance_scale=gs)
        sample, _ = sampler()

        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        write(join(target_dir, filename), x_hat.cpu().numpy(), sr)

    # Save adaptive guidance log
    if args.adaptive_guidance and guidance_log:
        log_path = join(target_dir, "_guidance_log.csv")
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "confidence", "pred_class", "w"])
            writer.writeheader()
            writer.writerows(guidance_log)
        confs = [r["confidence"] for r in guidance_log]
        ws = [r["w"] for r in guidance_log]
        print(f"\nAdaptive guidance stats:")
        print(f"  confidence: mean={sum(confs)/len(confs):.3f}, min={min(confs):.3f}, max={max(confs):.3f}")
        print(f"  w: mean={sum(ws)/len(ws):.3f}, min={min(ws):.3f}, max={max(ws):.3f}")
        print(f"  log saved to {log_path}")
