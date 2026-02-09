# NASE Fork Migration Guide

> 이 문서는 기존 SGMSE 기반 실험에서 NASE fork로 migration하기 위한 모든 정보를 담고 있습니다.
> Interspeech 2026 제출 목표 (D-18, 2/24 마감)

---

## 1. 연구 배경 및 목표

### 1.1 Original Goal
Noise-conditioned speech enhancement with CFG (Classifier-Free Guidance)를 통해 OOD (Out-of-Distribution) noise에서도 robust하게 동작하는 모델 개발.

### 1.2 Research Pivot (2026-02-05)
기존 scratch CNN encoder + CFG 접근이 실패하여, **NASE 기반 + Uncertainty-aware Adaptive Guidance**로 방향 전환.

### 1.3 핵심 아이디어
```
기존 NASE의 한계: OOD noise에서 misleading conditioning → 성능 하락
우리의 해결책: NC classifier confidence 또는 embedding distance로
             conditioning reliability를 추정하고, guidance scale w를 자동 조절
```

---

## 2. 기존 실험 결과 요약

### 2.1 성능 문제 발견

| Model | In-dist PESQ | In-dist SI-SDR | OOD PESQ | OOD SI-SDR |
|-------|--------------|----------------|----------|------------|
| **Paper pretrained** | **2.91** | **17.0** | - | - |
| Our SGMSE (N=50) | 1.84 | 12.9 | 1.19 | -1.3 |
| Our CFG p=0.2 (N=50) | 1.76 | 11.7 | 1.16 | -0.7 |

### 2.2 원인 분석

#### ❌ 학습 실패 원인: **48kHz vs 16kHz 문제**
```
- VoiceBank-DEMAND 데이터: 48kHz
- SGMSE STFT 파라미터: 16kHz용 (n_fft=510, hop_length=128)
- Training data_module.py: 리샘플링 없음!
- Inference enhancement.py: 16kHz로 리샘플링함

→ Training은 잘못된 spectrogram으로 학습, Inference는 정상
→ Paper pretrained는 16kHz 데이터로 학습되어 정상 작동
```

#### ❌ 기존 CNN Encoder 문제 (NASE 비교 분석)
| 요소 | NASE (잘 동작) | 우리 기존 (문제) |
|------|----------------|------------------|
| Noise Encoder | Pre-trained BEATs (768-dim) | 4-layer CNN from scratch (512-dim) |
| Encoder Supervision | NC loss (classification) | 없음 |
| Injection 방식 | Input addition | FiLM via time embedding |
| CFG | 없음 | p=0.2 dropout |

### 2.3 Training 설정 비교 (참고용)
| 항목 | Paper | 우리 |
|------|-------|------|
| Epochs/Steps | 160 epochs ≈ 58k steps | 58k steps ✅ |
| Batch | 4 GPU × 8 = 32 | 4 GPU × 8 = 32 ✅ |
| Learning rate | 1e-4 | 1e-4 ✅ |
| EMA decay | 0.999 | 0.999 ✅ |
| Backbone | ncsnpp | ncsnpp ✅ |
| **Data SR** | **16kHz** | **48kHz** ❌ |

---

## 3. 새로운 연구 방향: Uncertainty-aware Adaptive Guidance

### 3.1 Motivation
2025년 CFG 연구 트렌드: static guidance scale → adaptive/dynamic guidance

| Paper | Method | 우리와의 차이 |
|-------|--------|--------------|
| β-CFG (Feb 2025) | Timestep 기반 β-distribution | OOD 미고려 |
| Prompt-aware CFG (Sep 2025) | Prompt 복잡도 기반 | Noise embedding 특화 아님 |
| Feedback Guidance (Jun 2025) | Trajectory quality 자가평가 | 별도 evaluator 필요 |
| Dynamic CFG (Sep 2025) | Greedy search per timestep | Computational overhead |
| **Ours** | **Uncertainty 기반 (NC confidence / embedding distance)** | **OOD-aware, SE 특화** |

### 3.2 Proposed Method

```python
# Option 1: NC Confidence 기반
confidence = softmax(nc_logits).max()
w = confidence  # confident → w=1 (trust conditioning), uncertain → w→0 (fallback)

# Option 2: Embedding Distance 기반
dist = knn_distance(noise_emb, train_embeddings, k=10)
w = 1 / (1 + dist / tau)  # 가까우면 w→1, 멀면 w→0

# Option 3: Combined
w = alpha * confidence + (1-alpha) * (1 / (1 + dist/tau))
```

### 3.3 Updated Contributions

1. **Uncertainty-aware Adaptive Guidance**
   - NC classifier confidence나 embedding distance 기반으로 conditioning reliability 추정
   - Guidance scale w를 자동 조절 (수동 설정 불필요)

2. **OOD-Robust Speech Enhancement**
   - 별도의 OOD detector 없이, noise encoder의 uncertainty만으로 OOD 감지
   - Graceful degradation: OOD에서 자동으로 unconditional로 fallback

3. **Empirical Analysis**
   - 언제 noise conditioning이 도움/해가 되는지 분석
   - Uncertainty-performance correlation 분석

### 3.4 DiTSE와의 차별화
| | DiTSE (2025) | Ours |
|---|-------------|------|
| CFG 대상 | 모든 conditioning (WavLM 등) | Noise embedding만 |
| CFG 목적 | "conditioning을 더 잘 활용" | "OOD에서 graceful degradation" |
| Guidance scale | 학습 후 고정 | Inference-time adaptive |
| OOD 고려 | ❌ 없음 | ✅ 핵심 contribution |

---

## 4. 실험 계획

### 4.1 필수 실험

| Exp | Purpose | Method |
|-----|---------|--------|
| **E1** | NASE baseline 재현 | BEATs + NC loss, p=0 |
| **E2** | Adaptive guidance 효과 | Static w vs Adaptive w 비교 |
| **E3** | Uncertainty correlation | NC confidence / embedding distance 분석 |
| **E4** | Main comparison | SGMSE+ vs NASE vs Ours |

### 4.2 Dataset

**Training**: VoiceBank-DEMAND (⚠️ 반드시 16kHz로 리샘플링!)
```bash
python preprocessing/resample_to_16k.py \
    --input_dir ./data/voicebank-demand \
    --output_dir ./data/voicebank-demand-16k
```

**In-distribution Test**: VoiceBank-DEMAND test (824 files)

**OOD Test**:
- ESC-50 noise + clean speech @ 0dB SNR
- UrbanSound8K (추가 예정)

### 4.3 Evaluation Metrics
- PESQ (Perceptual Evaluation of Speech Quality)
- ESTOI (Extended Short-Time Objective Intelligibility)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)

---

## 5. NASE Repository 참고사항

### 5.1 NASE 핵심 구조
```
NASE GitHub: https://github.com/YUCHEN005/NASE
Paper: https://arxiv.org/abs/2307.08029
```

**NASE 주요 컴포넌트:**
- BEATs encoder (pretrained, 768-dim)
- NC loss (Noise Classification loss for encoder supervision)
- Input addition (noise embedding을 첫 layer feature에 더함)
- SGMSE+ backbone (ncsnpp 기반)

### 5.2 BEATs Checkpoint
```
BEATs Paper: https://arxiv.org/abs/2212.09058
Checkpoint: https://github.com/microsoft/unilm/tree/master/beats
```

### 5.3 NASE에 추가해야 할 것
1. **CFG Training**: `p_uncond` 확률로 noise embedding dropout
2. **Adaptive Guidance**: NC head confidence 기반 w 계산
3. **OOD Evaluation**: ESC-50, UrbanSound8K 테스트 파이프라인

---

## 6. 기존 코드 중 재사용 가능한 것

### 6.1 Evaluation Scripts
```
calc_metrics.py          # PESQ, ESTOI, SI-SDR 계산
scripts/eval_batch.py    # Batch evaluation
```

### 6.2 Data Preprocessing
```
preprocessing/resample_to_16k.py   # 48kHz → 16kHz 변환
preprocessing/create_ood_test.py   # OOD test mixture 생성 (있다면)
```

### 6.3 Visualization
```
visualization/visualize_eval_results.py  # 결과 시각화
```

### 6.4 Adaptive Guidance 구현 (새로 작성 필요)
```python
# model에 추가할 메서드
def compute_adaptive_w(self, noise_emb, nc_logits):
    """Compute adaptive guidance scale based on uncertainty."""
    # NC confidence
    confidence = F.softmax(nc_logits, dim=-1).max(dim=-1)[0]

    # w = confidence (simple version)
    # 또는 embedding distance 기반으로 확장
    return confidence

def enhance_with_adaptive_w(self, y, noise_ref, N=50):
    """Enhancement with adaptive guidance scale."""
    # 1. Extract noise embedding
    noise_emb, nc_logits = self.noise_encoder(noise_ref)

    # 2. Compute adaptive w
    w = self.compute_adaptive_w(noise_emb, nc_logits)

    # 3. Run CFG sampling with adaptive w
    # ...
```

---

## 7. 참고 논문 및 링크

### 7.1 Core Papers
| Paper | Venue | Link |
|-------|-------|------|
| SGMSE+ | TASLP 2023 | https://arxiv.org/abs/2203.17024 |
| NASE | Interspeech 2023 | https://arxiv.org/abs/2307.08029 |
| NADiffuSE | ASRU 2023 | https://arxiv.org/abs/2309.01212 |
| DiTSE | arXiv 2025 | https://arxiv.org/abs/2504.09381 |
| BEATs | ICML 2023 | https://arxiv.org/abs/2212.09058 |

### 7.2 CFG Improvements (2025)
| Paper | Link |
|-------|------|
| β-CFG | https://arxiv.org/abs/2502.10574 |
| Prompt-aware CFG | https://arxiv.org/abs/2509.22728 |
| Feedback Guidance | https://arxiv.org/abs/2506.06085 |
| Dynamic CFG | https://arxiv.org/abs/2509.16131 |

### 7.3 Repositories
| Repo | Link |
|------|------|
| SGMSE | https://github.com/sp-uhh/sgmse |
| NASE | https://github.com/YUCHEN005/NASE |
| BEATs | https://github.com/microsoft/unilm/tree/master/beats |

---

## 8. 일정 (D-18, 2/24 제출)

### Week 1 (2/6-2/12)
- [ ] 2/6: 멘토님 미팅, 방향 확정
- [ ] 2/6-2/7: NASE fork, 환경 설정
- [ ] 2/7: **데이터 16kHz 리샘플링**
- [ ] 2/8-2/10: NASE baseline 학습 (p=0)
- [ ] 2/11-2/12: CFG 추가 (p=0.2) 학습

### Week 2 (2/13-2/19)
- [ ] 2/13-2/15: Main comparison 실험 (E1, E4)
- [ ] 2/16-2/17: Adaptive guidance 구현 및 실험 (E2)
- [ ] 2/18-2/19: Uncertainty analysis (E3)

### Week 3 (2/20-2/24)
- [ ] 2/20-2/22: 논문 작성 (Results, Discussion)
- [ ] 2/23: Final review
- [ ] 2/24: **제출**

---

## 9. Checklist for NASE Fork

### 9.1 환경 설정
- [ ] NASE repository fork
- [ ] Dependencies 설치
- [ ] BEATs checkpoint 다운로드
- [ ] VoiceBank-DEMAND 16kHz 변환

### 9.2 코드 수정
- [ ] CFG dropout 추가 (`p_uncond` 파라미터)
- [ ] Adaptive guidance 구현 (NC confidence 기반)
- [ ] OOD evaluation 파이프라인 추가
- [ ] Visualization scripts 연동

### 9.3 실험
- [ ] NASE baseline 재현 (In-dist PESQ > 2.9 목표)
- [ ] NASE의 OOD misleading 문제 확인
- [ ] Adaptive guidance 효과 검증
- [ ] Uncertainty-performance correlation 분석

---

## 10. 멘토님 미팅 요약 (2026-02-06 예정)

### 논의할 내용
1. **48kHz 데이터 문제** - 이게 학습 실패의 핵심 원인
2. **NASE fork로 재시작** - 기존 코드 버리고 새로 시작
3. **Uncertainty-aware Adaptive Guidance** - Novelty 충분한지
4. **일정 현실성** - 18일 안에 가능한지

### 질문
1. Adaptive guidance가 DiTSE 대비 충분한 novelty인가?
2. 만약 가설이 틀리면 (adaptive가 효과 없으면) Plan B는?
3. 최소 viable paper를 위한 필수 실험 우선순위는?

---

*Created: 2026-02-06*
*For migration from: sgmse (sp-uhh fork)*
*To: NASE fork with Uncertainty-aware Adaptive Guidance*
