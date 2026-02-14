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
    # Adaptive guidance args (CFG-based, legacy)
    parser.add_argument("--adaptive_guidance", action='store_true', help="Enable adaptive CFG guidance (overrides --guidance_scale)")
    parser.add_argument("--w_max", type=float, default=1.0, help="Max guidance scale for adaptive mode")
    parser.add_argument("--guidance_mapping", type=str, default="linear", choices=("linear", "scaled", "binary"), help="Confidence-to-w mapping")
    parser.add_argument("--distance_method", type=str, default="confidence", choices=("confidence", "knn", "prototype"), help="Adaptive w method")
    parser.add_argument("--ref_embeddings", type=str, default=None, help="Path to reference embeddings .pt file")
    parser.add_argument("--k_neighbors", type=int, default=10, help="k for k-NN distance")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for distance-to-alpha mapping")
    parser.add_argument("--use_raw", action='store_true', help="Use raw encoder features for distance")
    # Embedding scaling args (new approach, no CFG needed)
    parser.add_argument("--noise_scale", type=float, default=None, help="Static noise embedding scale (0.0=no conditioning, 1.0=normal). Overrides adaptive.")
    parser.add_argument("--adaptive_scaling", action='store_true', help="Enable distance-based adaptive noise embedding scaling (no CFG)")
    parser.add_argument("--scaling_method", type=str, default="knn", choices=("knn", "prototype"), help="Distance method for adaptive scaling")
    parser.add_argument("--encoder_type", type=str, default="beats", choices=("beats", "wavlm", "panns"), help="Noise encoder type (must match training)")
    # Multi-degradation adaptive args
    parser.add_argument("--multi_degradation", action='store_true', help="Enable multi-degradation adaptive inference")
    parser.add_argument("--static_noise_w", type=float, default=None, help="Static noise branch weight (overrides adaptive)")
    parser.add_argument("--static_reverb_w", type=float, default=None, help="Static reverb branch weight (overrides adaptive)")
    parser.add_argument("--static_distort_w", type=float, default=None, help="Static distortion branch weight (overrides adaptive)")
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
    load_kwargs = dict(
        base_dir='',
        batch_size=16,
        num_workers=0,
        pretrain_class_model=pretrain_class_model,
        encoder_type=args.encoder_type,
        kwargs=dict(gpu=False),
    )
    if args.multi_degradation:
        load_kwargs['multi_degradation'] = True
    model = ScoreModel.load_from_checkpoint(checkpoint_file, **load_kwargs)
    model.eval(no_ema=False)
    model.cuda()

    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))

    # Load reference embeddings for distance-based methods
    ref_embeddings = None
    needs_ref = (args.adaptive_guidance and args.distance_method in ("knn", "prototype")) or args.adaptive_scaling
    if needs_ref:
        if args.ref_embeddings is None:
            raise ValueError("--ref_embeddings required for distance-based methods")
        ref_embeddings = torch.load(args.ref_embeddings, map_location="cuda", weights_only=True)
        print(f"Loaded reference embeddings: {args.ref_embeddings} — shape {ref_embeddings.shape}")

    # Determine mode
    if args.multi_degradation:
        if args.static_noise_w is not None:
            mode = f"Multi-deg static (n={args.static_noise_w}, r={args.static_reverb_w}, d={args.static_distort_w})"
        else:
            mode = "Multi-deg adaptive"
    elif args.noise_scale is not None:
        mode = f"Static noise_scale={args.noise_scale:.2f}"
    elif args.adaptive_scaling:
        mode = f"Adaptive scaling ({args.scaling_method}, tau={args.tau})"
    elif args.adaptive_guidance:
        mode = f"Adaptive CFG ({args.distance_method}, w_max={args.w_max}, tau={args.tau})"
    elif args.guidance_scale is not None:
        mode = "CFG w={:.1f}".format(args.guidance_scale)
    else:
        mode = "baseline (no CFG)"
    print(f'start inference! mode={mode}, N={N}, files={len(noisy_files)}')

    # Per-file log
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

        # --- Multi-degradation adaptive mode ---
        multi_weights = None
        if args.multi_degradation:
            if args.static_noise_w is not None:
                # Static per-branch weights
                noise_w = args.static_noise_w
                reverb_w = args.static_reverb_w if args.static_reverb_w is not None else 1.0
                distort_w = args.static_distort_w if args.static_distort_w is not None else 1.0
                info = {"noise_w": noise_w, "reverb_w": reverb_w, "distort_w": distort_w}
            else:
                # Adaptive per-branch weights from encoder predictions
                noise_w, reverb_w, distort_w, info = model.compute_multi_adaptive_weights(
                    y_wav.cuda())
            multi_weights = (noise_w, reverb_w, distort_w)
            gs = None  # no legacy CFG
            guidance_log.append({
                "filename": filename,
                "distance": 0.0,
                "alpha": 1.0,
                "w": 0.0,
                "noise_w": round(noise_w, 4),
                "reverb_w": round(reverb_w, 4),
                "distort_w": round(distort_w, 4),
            })

        # --- Embedding Scaling mode (new) ---
        elif args.noise_scale is not None:
            # Static noise scale
            model.noise_scale = args.noise_scale
            gs = None  # no CFG
            guidance_log.append({"filename": filename, "distance": 0.0, "alpha": args.noise_scale, "w": 0.0})

        elif args.adaptive_scaling:
            # Distance-based adaptive noise scaling (no CFG)
            if args.scaling_method == "knn":
                distance, alpha_tensor = model.compute_embedding_distance(
                    y_wav.cuda(), ref_embeddings, k=args.k_neighbors, tau=args.tau,
                    use_raw=args.use_raw)
            else:  # prototype
                distance, alpha_tensor = model.compute_prototype_distance(
                    y_wav.cuda(), ref_embeddings, tau=args.tau,
                    use_raw=args.use_raw)
            dist_val = distance.item()
            # Map w∈[-1,1] to alpha∈[0,1]: alpha = (w+1)/2
            alpha = max(0.0, min(1.0, (alpha_tensor.item() + 1.0) / 2.0))
            model.noise_scale = alpha
            gs = None  # no CFG
            guidance_log.append({"filename": filename, "distance": round(dist_val, 4), "alpha": round(alpha, 4), "w": 0.0})

        # --- Legacy CFG adaptive mode ---
        elif args.adaptive_guidance:
            model.noise_scale = 1.0  # reset
            if args.distance_method == "confidence":
                confidence, logits = model.compute_nc_confidence(y_wav.cuda())
                conf = confidence.item()
                pred_class = torch.argmax(logits, dim=-1).item()
                if args.guidance_mapping == "linear":
                    w = args.w_max * (2 * conf - 1)
                elif args.guidance_mapping == "scaled":
                    w = args.w_max * conf
                elif args.guidance_mapping == "binary":
                    w = args.w_max if conf > 0.5 else 0.0
                guidance_log.append({"filename": filename, "distance": 0.0, "alpha": 1.0, "w": round(w, 4)})
            elif args.distance_method == "knn":
                distance, w_tensor = model.compute_embedding_distance(
                    y_wav.cuda(), ref_embeddings, k=args.k_neighbors, tau=args.tau,
                    use_raw=args.use_raw)
                w = w_tensor.item() * args.w_max
                guidance_log.append({"filename": filename, "distance": round(distance.item(), 4), "alpha": 1.0, "w": round(w, 4)})
            elif args.distance_method == "prototype":
                distance, w_tensor = model.compute_prototype_distance(
                    y_wav.cuda(), ref_embeddings, tau=args.tau,
                    use_raw=args.use_raw)
                w = w_tensor.item() * args.w_max
                guidance_log.append({"filename": filename, "distance": round(distance.item(), 4), "alpha": 1.0, "w": round(w, 4)})
            gs = w if abs(w) > 1e-6 else None
        else:
            model.noise_scale = 1.0  # reset
            gs = args.guidance_scale

        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', corrector_cls, Y.cuda(), y_wav.cuda(), N=N,
            corrector_steps=corrector_steps, snr=snr,
            guidance_scale=gs,
            multi_adaptive_weights=multi_weights)
        sample, _ = sampler()

        # Backward transform in time domain
        x_hat = model.to_audio(sample.squeeze(), T_orig)

        # Renormalize
        x_hat = x_hat * norm_factor

        # Write enhanced wav file
        write(join(target_dir, filename), x_hat.cpu().numpy(), sr)

    # Save log
    if guidance_log:
        log_path = join(target_dir, "_guidance_log.csv")
        if args.multi_degradation:
            fieldnames = ["filename", "distance", "alpha", "w", "noise_w", "reverb_w", "distort_w"]
        else:
            fieldnames = ["filename", "distance", "alpha", "w"]
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(guidance_log)
        if args.multi_degradation:
            nws = [r["noise_w"] for r in guidance_log]
            rws = [r["reverb_w"] for r in guidance_log]
            dws = [r["distort_w"] for r in guidance_log]
            print(f"\nMulti-degradation adaptive weights:")
            print(f"  noise_w:   mean={sum(nws)/len(nws):.3f}, min={min(nws):.3f}, max={max(nws):.3f}")
            print(f"  reverb_w:  mean={sum(rws)/len(rws):.3f}, min={min(rws):.3f}, max={max(rws):.3f}")
            print(f"  distort_w: mean={sum(dws)/len(dws):.3f}, min={min(dws):.3f}, max={max(dws):.3f}")
        elif args.adaptive_scaling or args.noise_scale is not None:
            alphas = [r["alpha"] for r in guidance_log]
            dists = [r["distance"] for r in guidance_log]
            print(f"\nEmbedding scaling stats:")
            print(f"  alpha: mean={sum(alphas)/len(alphas):.3f}, min={min(alphas):.3f}, max={max(alphas):.3f}")
            if any(d > 0 for d in dists):
                print(f"  distance: mean={sum(dists)/len(dists):.3f}, min={min(dists):.3f}, max={max(dists):.3f}")
        elif args.adaptive_guidance:
            ws = [r["w"] for r in guidance_log]
            dists = [r["distance"] for r in guidance_log]
            print(f"\nAdaptive guidance stats ({args.distance_method}):")
            if any(d > 0 for d in dists):
                print(f"  distance: mean={sum(dists)/len(dists):.3f}, min={min(dists):.3f}, max={max(dists):.3f}")
            print(f"  w: mean={sum(ws)/len(ws):.3f}, min={min(ws):.3f}, max={max(ws):.3f}")
        print(f"  log saved to {log_path}")
