from os.path import join
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import torch

from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std

# Fix for PyTorch 2.6+ weights_only=True default breaking fairseq/UTMOS
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the original test data (must have subdirectories clean/ and noisy/)')
    parser.add_argument("--test_set", type=str, default='noisy')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--utmos", action='store_true', help='Also compute UTMOS (neural MOS prediction, requires pip install utmos)')
    args = parser.parse_args()

    test_dir = args.test_dir
    clean_dir = join(test_dir, "clean/")
    noisy_dir = join(test_dir, args.test_set)
    enhanced_dir = args.enhanced_dir

    # Load UTMOS model if requested
    utmos_model = None
    if args.utmos:
        import utmos
        utmos_model = utmos.Score()
        print("UTMOS model loaded")

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": []}
    if args.utmos:
        data["utmos"] = []
    sr = 16000

    # Evaluate standard metrics
    noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.split('/')[-1]
        x, _ = read(join(clean_dir, filename))
        y, _ = read(noisy_file)
        x_method, _ = read(join(enhanced_dir, filename))

        # Truncate to same length (some models slightly change length)
        min_len = min(len(x), len(x_method), len(y))
        x = x[:min_len]
        x_method = x_method[:min_len]
        y = y[:min_len]
        n = y - x

        data["filename"].append(filename)
        data["pesq"].append(pesq(sr, x, x_method, 'wb'))
        data["estoi"].append(stoi(x, x_method, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
        data["si_sir"].append(energy_ratios(x_method, x, n)[1])
        data["si_sar"].append(energy_ratios(x_method, x, n)[2])

        if utmos_model is not None:
            enhanced_path = join(enhanced_dir, filename)
            data["utmos"].append(utmos_model.calculate_wav_file(enhanced_path))

    # Save results as DataFrame
    df = pd.DataFrame(data)

    # Print results
    print(enhanced_dir)
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))
    if args.utmos:
        print("UTMOS: {:.3f} ± {:.3f}".format(*mean_std(df["utmos"].to_numpy())))

    # Save DataFrame as csv file
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)
