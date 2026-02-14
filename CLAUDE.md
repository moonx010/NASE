# NASE: Multi-Degradation Adaptive Speech Enhancement

> Interspeech 2026 제출 (D-10, 2/24 마감)
> 기존 SGMSE→NASE migration 완료, Multi-Degradation Adaptive SE 구현 완료 (2/14)

---

## 1. 연구 개요

### 1.1 한줄 요약
Diffusion 기반 SE에서 noise/reverb/distortion 복합 degradation을 각각 독립적으로 conditioning하고,
encoder prediction confidence 기반으로 per-degradation adaptive guidance를 수행하는 모델.

### 1.2 연구 진화 경로
```
Phase 0 (2/6): SGMSE + CNN encoder + CFG → 실패 (48kHz 문제 + shallow injection)
Phase 1 (2/7): NASE fork + BEATs + input addition → CFG 무효 확인
Phase 2 (2/14): WavLM + 3 heads + temb injection + per-degradation adaptive → 현재
```

### 1.3 핵심 아이디어
```
문제:
  1. Input addition은 shallow (1 injection point) → CFG 작동 안 함
  2. Single degradation conditioning → 복합 환경에서 부정확
  3. OOD 상황에서 blind conditioning → 성능 악화

해결:
  1. temb injection → ~37개 ResBlock 전체에 deep conditioning
  2. 3-branch encoding (noise/reverb/distortion) → per-degradation 독립 제어
  3. Encoder prediction confidence → adaptive weight 자동 조절
```

---

## 2. Architecture

### 2.1 전체 구조
```
noisy_wav → WavLM Encoder → (B, T, 768)
                │
                ├─→ noise_head(768→11)  → 11-class CE loss
                ├─→ reverb_head(768→256→1) → T60 MSE loss
                └─→ distort_head(768→256→1) → intensity MSE loss
                │
                └─→ post_cnn(768→256) → mean pool → (B, 256)
                     │
                     ├─→ noise_proj(256→128)  × dropout_mask
                     ├─→ reverb_proj(256→128)  × dropout_mask
                     └─→ distort_proj(256→128) × dropout_mask
                          │
                          concat → (B, 384) → cond_to_temb(384→512→512)
                                                    │
STFT(noisy) → ncsnpp backbone ← temb(timestep) + extra_cond(degradation)
     │                              ↑ 모든 ~37 ResBlock에 주입
     └─→ ISTFT → enhanced speech
```

### 2.2 핵심 모듈 위치

| Module | File | Line(approx) |
|--------|------|------|
| WavLMEncoder (3 heads) | `sgmse/encoders.py` | WavLMEncoder class |
| 3-branch projection | `sgmse/model.py` | ScoreModel.__init__ |
| forward_train_multi | `sgmse/model.py` | forward_train_multi() |
| temb injection | `sgmse/backbones/ncsnpp.py:290` | `temb = temb + extra_cond` |
| Multi-task loss | `sgmse/model.py` | _step_train_multi() |
| Adaptive inference | `sgmse/model.py` | forward_multi_adaptive() |
| Specs_multi_label | `sgmse/data_module.py` | CSV-based multi-label dataset |
| Data generation | `preprocessing/create_multi_degradation.py` | 복합 degradation 생성 |

### 2.3 Injection Method 비교

| 방법 | Injection Points | CFG 가능 | 구현 |
|------|-----------------|---------|------|
| Input addition (NASE) | 1 | X (실험 확인) | 기존 코드 |
| Cross-attention | 1 | △ (미검증) | 기존 코드 |
| **temb injection (Ours)** | **~37** | **O** | **ncsnpp 1줄** |
| AdaLN-Zero (DiTSE) | 전체 layer | O (검증됨) | backbone 교체 필요 |

---

## 3. 코드 구조

### 3.1 주요 파일

```
NASE/
├── train.py                  # 학습 진입점
├── enhancement.py            # 추론/평가 진입점
├── calc_metrics.py           # PESQ, ESTOI, SI-SDR 계산
├── sgmse/
│   ├── model.py              # ScoreModel (핵심: 3-branch, multi-task loss)
│   ├── encoders.py           # BEATs/WavLM/PANNs encoder
│   ├── data_module.py        # Dataset classes (Specs, Specs_noise_label, Specs_multi_label)
│   ├── backbones/
│   │   └── ncsnpp.py         # U-Net backbone (temb injection 여기)
│   ├── sampling.py           # PC sampler
│   ├── sdes.py               # SDE definitions
│   └── BEATs.py              # BEATs model
├── preprocessing/
│   ├── create_multi_degradation.py  # 복합 degradation 데이터 생성
│   ├── create_ood_test.py           # OOD 테스트 생성
│   └── resample_to_16k.py          # 48kHz→16kHz
└── scripts/
    └── eval_batch.py          # 배치 평가
```

### 3.2 핵심 Flag

| Flag | 위치 | 설명 |
|------|------|------|
| `--multi_degradation` | SpecsDataModule | 3-head + temb injection 모드 활성화 |
| `--encoder_type wavlm` | train.py / enhancement.py | WavLM encoder 사용 (multi_deg 필수) |
| `--p_uncond 0.1` | ScoreModel | Per-branch dropout 확률 |
| `--static_noise_w` | enhancement.py | 정적 noise branch weight |
| `--static_reverb_w` | enhancement.py | 정적 reverb branch weight |
| `--static_distort_w` | enhancement.py | 정적 distort branch weight |

### 3.3 학습 명령어 템플릿

```bash
python train.py --backbone ncsnpp --sde ouve \
    --encoder_type wavlm --multi_degradation \
    --pretrain_class_model dummy \
    --base_dir /path/to/multi_degradation_16k \
    --gpus 4 --batch_size 4 --p_uncond 0.1 \
    --wandb_name multi-deg-wavlm-v1 \
    --max_epochs 160
```

### 3.4 평가 명령어 템플릿

```bash
# Baseline (no adaptation)
python enhancement.py --multi_degradation \
    --ckpt checkpoint.ckpt --pretrain_class_model dummy \
    --encoder_type wavlm \
    --test_dir test_dir --enhanced_dir enhanced/baseline \
    --N 50

# Adaptive (per-degradation automatic)
python enhancement.py --multi_degradation \
    --ckpt checkpoint.ckpt --pretrain_class_model dummy \
    --encoder_type wavlm \
    --test_dir test_dir --enhanced_dir enhanced/adaptive \
    --N 50
# (multi_degradation 모드에서는 기본적으로 w=1.0 전체 사용)

# Static ablation
python enhancement.py --multi_degradation \
    --static_noise_w 1.0 --static_reverb_w 0.0 --static_distort_w 0.0 \
    --ckpt checkpoint.ckpt --pretrain_class_model dummy \
    --encoder_type wavlm \
    --test_dir test_dir --enhanced_dir enhanced/noise-only \
    --N 50
```

---

## 4. 데이터

### 4.1 Training Data

| Dataset | Purpose | Size | Format |
|---------|---------|------|--------|
| VoiceBank-DEMAND 16kHz | 기존 noise-only | 11,572 train | clean/ + noisy/ + noise_label.txt |
| Multi-Degradation 16kHz | 복합 degradation | ~30,000 train | clean/ + noisy/ + labels.csv |

### 4.2 labels.csv Format
```csv
filename,noise_type,snr,reverb_t60,distort_intensity
p236_002_n.wav,babble,15.0,0.0,0.0
p236_002_nr.wav,babble,10.0,0.65,0.0
p236_002_nrd.wav,cafeteria,5.0,0.8,0.7
p236_003_r.wav,none,0.0,0.55,0.0
p236_003_d.wav,none,0.0,0.0,0.6
```

### 4.3 Noise Classes (11)
```
0:babble, 1:cafeteria, 2:car, 3:kitchen, 4:meeting,
5:metro, 6:restaurant, 7:ssn, 8:station, 9:traffic, 10:none
```

### 4.4 Degradation Combinations
| Suffix | Combination | Fields |
|--------|-------------|--------|
| `_n` | noise only | noise_type, snr |
| `_r` | reverb only | reverb_t60 |
| `_d` | distort only | distort_intensity |
| `_nr` | noise + reverb | noise_type, snr, reverb_t60 |
| `_nd` | noise + distort | noise_type, snr, distort_intensity |
| `_nrd` | noise + reverb + distort | all |

---

## 5. 기존 실험 결과 요약

### 5.1 Phase 0: NASE Baseline

| Model | In-dist PESQ | In-dist SI-SDR | OOD PESQ |
|-------|-------------|----------------|----------|
| Paper pretrained | 2.91 | 17.0 | - |
| Our retrain (48kHz, 버그) | 1.84 | 12.9 | 1.19 |
| Our + CFG p=0.2 | 1.76 | 11.7 | 1.16 |

**핵심 발견:**
1. 48kHz 데이터 사용이 학습 실패 원인 (해결됨: 16kHz)
2. Input addition에서 CFG 무효 (해결됨: temb injection)

### 5.2 Phase 1: Adaptive Guidance (Single-Degradation)
- Embedding scaling (alpha sweep): 진행 중
- Confidence-based scaling: 구현 완료, 평가 필요

### 5.3 Phase 2: Multi-Degradation (Current)
- 코드 구현 완료 (2/14)
- 데이터 생성 → 학습 → 평가 순서로 진행 예정

---

## 6. 실험 계획

> 상세 계획: `memory/research_plan.md`
> 실험 로그: `memory/experiment_log.md`

### 6.1 필수 실험 (P0)

| ID | Experiment | Status |
|----|-----------|--------|
| E2-0 | Multi-deg 데이터 생성 | 코드 완료 |
| E2-1 | WavLM + multi-deg 학습 | 코드 완료, 실행 필요 |
| E2-2 | In-dist 평가 | 대기 |
| E2-5 | OOD 평가 | 대기 |
| E7 | Main comparison table | 대기 |

### 6.2 선택 실험 (P1, 시간 여유시)

| ID | Experiment | Purpose |
|----|-----------|---------|
| E3 | Injection method ablation | temb vs input addition |
| E4 | Adaptive method ablation | per-branch vs global |
| E5 | Cross-degradation analysis | Per-branch weight 시각화 |

### 6.3 Metrics
- PESQ, ESTOI, SI-SDR (speech quality)
- NC Accuracy (학습 모니터링)
- Per-branch weight distribution (해석 가능성)

---

## 7. Contributions (논문용)

1. **Multi-Degradation Conditioning via temb Injection**
   - 3-branch (noise/reverb/distortion) encoding
   - Timestep embedding injection → ~37 ResBlock 전체에 deep conditioning

2. **Per-Degradation Adaptive Guidance**
   - 각 head의 prediction confidence → per-branch weight
   - OOD 자동 감지 + graceful degradation

3. **Empirical Analysis**
   - Cross-degradation evaluation
   - Per-branch weight distribution analysis

---

## 8. 서버 & 환경

| Item | Value |
|------|-------|
| Server 1 | 159-145 (8x RTX 3090) |
| Server 2 | 159-67 (7x RTX 3090) |
| NAS | `/home/nas4_user/kyudanjung/seokhoonmoon/` |
| Conda env | `sgmse` (torch 2.8.0+cu128, PL 2.5.5) |
| Data (16kHz) | NAS 또는 서버 로컬 |
| BEATs ckpt | NAS |
| GitHub | `git@github.com:moonx010/NASE.git` |

### 주의사항
- 데이터는 반드시 **16kHz** (48kHz 사용 금지!)
- RTX 3090: batch_size=4 max (WavLM + multi-deg는 더 작을 수 있음)
- tmux에서 항상 `conda activate sgmse`
- PL 2.x: `Trainer.add_argparse_args`, `from_argparse_args` 없음
- `--multi_degradation`은 `SpecsDataModule.add_argparse_args`에서만 등록

---

## 9. 참고 논문

### Core
| Paper | Venue | Key |
|-------|-------|-----|
| SGMSE+ | TASLP 2023 | Diffusion SE baseline |
| NASE | Interspeech 2023 | Noise-aware conditioning |
| NADiffuSE | ASRU 2023 | Noise-aware diffusion |
| DiTSE | arXiv 2025 | DiT + AdaLN-Zero + CFG |
| BEATs | ICML 2023 | Audio SSL encoder |
| WavLM | JSTSP 2022 | Speech SSL encoder |

### Adaptive CFG
| Paper | Key |
|-------|-----|
| beta-CFG (2025) | Timestep-based beta |
| Prompt-aware CFG (2025) | Prompt complexity |
| Dynamic CFG (2025) | Per-timestep search |

---

## 10. Timeline (D-10)

| Date | Task |
|------|------|
| 2/14 | 코드 구현 완료 ✅ |
| 2/15 | 데이터 생성 + 학습 시작 |
| 2/16-18 | 학습 (160ep) |
| 2/18 | 중간 eval (40ep ckpt) |
| 2/19 | Full eval (E2, E6) |
| 2/20 | Ablation (시간 여유시) |
| 2/21-22 | 논문 작성 |
| 2/23 | Final review |
| 2/24 | **Interspeech 제출** |

---

*Last updated: 2026-02-14*
*Phase: Multi-Degradation Adaptive SE 구현 완료, 학습 대기*
