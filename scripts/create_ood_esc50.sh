#!/usr/bin/env bash
# OOD test mixture: ESC-50 noise + clean speech @ 0dB SNR
D="/home/nas4_user/kyudanjung/seokhoonmoon/data"
python preprocessing/create_ood_test.py --clean_dir $D/voicebank-demand-16k/test/clean --noise_dir $D/ESC-50-master/audio --output_dir $D/ood_test/esc50 --esc50_meta $D/ESC-50-master/meta/esc50.csv --snr 0 --max_files 200
