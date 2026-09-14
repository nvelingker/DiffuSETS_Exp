# Clean DiffuSETS full test comparison (2026-09-14)

> **Diagnostic-only result:** this evaluation exposed a hyperparameter-source
> error in the local v1 diffusion run. The v1 U-Net and all generations below
> are quarantined from new comparisons. The metrics remain a reproducible
> record of the failure and must not be reported as the corrected DiffuSETS
> result. See
> [DIFFUSION_UNDERTRAINING_AUDIT_20260914.md](DIFFUSION_UNDERTRAINING_AUDIT_20260914.md).

This is the registered same-scorer comparison of the local patient-disjoint
DiffuSETS seed-2026 suite against ECGDiff epoch 16 and the local SE-Diff v3
epoch-195 lock. No author-released DiffuSETS checkpoint or saved generation
is used in any current score.

## Direct result

The clean DiffuSETS diffusion checkpoint performs substantially worse than both
comparators on both panels. On MIMIC, raw MAE is 0.46795 mV and QRS F1 is
0.54063, versus 0.12320/0.81272 for ECGDiff and 0.11137/0.78214 for
SE-Diff. On PTB-XL, raw MAE is 0.47626 mV and QRS F1 is 0.53613, versus
0.15051/0.77686 and 0.14366/0.75389. Clean-CLIP64 manifold recall is only
0.0037 on MIMIC and 0.0080 on PTB-XL. These independent failures support a
broad generation-quality problem. Lead-order checks pass: the saved outputs are
canonicalized before all scorers, and frontal-lead identity is evaluated after
that conversion.

All entries use one draw per condition. Lower is better unless a row says
otherwise; bold marks the best literal value in a row. Aligned waveform values
are secondary diagnostics after per-lead centering and one shared circular lag
within +/-1 second.

## MIMIC full shared test panel: 2,149 records / 419 patients

### Waveform fidelity

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| Raw paired MAE (mV) | 0.12320 | 0.46795 | **0.11137** |
| Raw paired MSE (mV^2) | 0.07157 | 0.70952 | **0.05605** |
| Raw global RMSE (mV) | 0.26752 | 0.84233 | **0.23674** |
| Raw mean-record RMSE (mV) | 0.24932 | 0.83182 | **0.21836** |
| Local per-lead-PTP NRMSE | 0.19867 | 0.73747 | **0.17202** |
| Raw mean per-lead Pearson r | **0.10502** | 0.00990 | 0.08902 |
| Centered/aligned MAE, +/-1 s (mV) | 0.10737 | 0.45730 | **0.10107** |
| Centered/aligned MSE, +/-1 s (mV^2) | 0.05423 | 0.66526 | **0.04424** |
| Centered/aligned global RMSE (mV) | 0.23288 | 0.81563 | **0.21032** |
| Centered/aligned mean-record RMSE (mV) | 0.21653 | 0.80591 | **0.19541** |
| Centered/aligned Pearson r | **0.29687** | 0.08387 | 0.23844 |
| Alignment-boundary fraction (%) | 1.443 | 1.443 | **1.396** |

### Rhythm, QRS, and physical consistency

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| HR MAE to conditioning HR (bpm) | **9.18** | 33.88 | 18.53 |
| HR MAE to paired-real HR (bpm) | **15.23** | 30.70 | 21.10 |
| Within 10 bpm of conditioning HR (%) | **82.13** | 13.08 | 68.26 |
| Within 10 bpm of paired-real HR (%) | **69.47** | 17.12 | 60.07 |
| Beat-count MAE | **2.403** | 4.713 | 3.332 |
| Mean-RR MAE (ms) | **110.76** | 217.79 | 141.48 |
| RR Wasserstein (ms) | **129.11** | 299.50 | 159.34 |
| SDNN MAE (ms) | **64.97** | 222.72 | 69.95 |
| QRS precision | **0.83256** | 0.50782 | 0.76275 |
| QRS recall | 0.79380 | 0.57798 | **0.80254** |
| QRS F1 | **0.81272** | 0.54063 | 0.78214 |
| Matched-QRS timing MAE (ms) | **39.06** | 50.25 | 39.93 |
| Generated QRS detection coverage | **1.000** | **1.000** | **1.000** |
| Generated frontal-identity RMSE (mV) | **0.00281** | 0.06598 | 0.00479 |
| Paired absolute frontal-identity error (mV) | 0.00403 | 0.06006 | **0.00374** |
| Records exceeding 10 mV (%) | **0.000** | 13.076 | **0.000** |

### ECGDeli morphology

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| PR interval MAE (ms) | **18.229** | 33.529 | 21.322 |
| QRS duration MAE (ms) | **9.115** | 16.113 | 11.393 |
| QT interval MAE (ms) | **20.508** | 51.107 | 21.647 |
| QTc Fridericia MAE (ms) | **22.340** | 54.035 | 23.556 |
| ST at J+60 MAE (mV) | **0.01604** | 0.04023 | 0.01665 |
| P-wave duration MAE (ms) | **9.766** | 12.370 | 10.417 |
| T-wave duration MAE (ms) | **10.742** | 30.273 | 11.230 |
| Generated all-metric coverage (%) | **100.000** | 99.721 | 99.860 |
| Paired all-metric coverage (%) | **99.814** | 99.535 | 99.674 |

Morphology rows are median absolute paired-record errors conditional on
successful delineation; the fixed-denominator coverage rows expose every
failed record. This local canonical ECGDeli protocol is shared across the
three models and is not an author-paper evaluator reconstruction.

### DiffuSETS-clean-CLIP64 seed2026 sensitivity

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| FID | **5.072** | 16060.413 | 37.605 |
| Real contiguous-half FID | 1.635 | 1.635 | 1.635 |
| Manifold precision, k=3 | **0.9353** | 0.7613 | 0.9000 |
| Manifold recall, k=3 | **0.8888** | 0.0037 | 0.8241 |
| Manifold F1, k=3 | **0.9115** | 0.0074 | 0.8604 |
| ECG-text cosine | **0.8362** | 0.6834 | 0.8298 |
| Reference ECG-text cosine | 0.8426 | 0.8426 | 0.8426 |
| rCLIP ratio of means | **0.9924** | 0.8111 | 0.9848 |
| Legacy rFID, generated/real-split | **3.1023** | 9824.0426 | 23.0029 |
| ICLR rFID, real-split/sum | **0.2438** | 0.0001 | 0.0417 |

### Patient-bootstrap uncertainty

| Model | Raw MAE 95% CI | Raw MSE 95% CI | Aligned MAE 95% CI | HR MAE to real 95% CI | Mean-record QRS F1 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| ECGDiff e16 | [0.11662, 0.12997] | [0.06487, 0.07865] | [0.10118, 0.11317] | [13.23430, 17.39327] | [0.79982, 0.83692] |
| DiffuSETS clean e200 | [0.46060, 0.47520] | [0.69478, 0.72425] | [0.45085, 0.46351] | [29.30411, 32.09638] | [0.52891, 0.53931] |
| SE-Diff v3 e195 | [0.10607, 0.11716] | [0.05075, 0.06174] | [0.09572, 0.10646] | [19.21003, 23.11406] | [0.77253, 0.80374] |

The scorer intervals use 1,000 patient-clustered replicates (seed
20260903). The paired clean-minus-baseline intervals below use 10,000
patient-clustered replicates (seed 20260914). Positive error deltas mean
clean DiffuSETS is worse; negative QRS-F1 deltas mean it is worse.

| Baseline | Raw MAE delta [95% CI] | Aligned MAE delta [95% CI] | HR-to-real MAE delta [95% CI] | Mean-record QRS F1 delta [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| ECGDiff e16 | 0.34476 [0.34091, 0.34862] | 0.34993 [0.34594, 0.35390] | 15.47 [12.72, 18.00] | -0.28415 [-0.30478, -0.26347] |
| SE-Diff v3 e195 | 0.35658 [0.35243, 0.36060] | 0.35623 [0.35250, 0.35994] | 9.60 [7.01, 12.15] | -0.25406 [-0.27172, -0.23602] |

## Fixed PTB-XL OOD panel: 1,000 records / 991 patients

### Waveform fidelity

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| Raw paired MAE (mV) | 0.15051 | 0.47626 | **0.14366** |
| Raw paired MSE (mV^2) | 0.09484 | 0.73853 | **0.08708** |
| Raw global RMSE (mV) | 0.30796 | 0.85938 | **0.29509** |
| Raw mean-record RMSE (mV) | 0.27145 | 0.84408 | **0.25679** |
| Local per-lead-PTP NRMSE | 0.18100 | 0.61967 | **0.16691** |
| Raw mean per-lead Pearson r | -0.00006 | -0.00090 | **0.00047** |
| Centered/aligned MAE, +/-1 s (mV) | 0.12340 | 0.46401 | **0.12074** |
| Centered/aligned MSE, +/-1 s (mV^2) | 0.06329 | 0.67994 | **0.05911** |
| Centered/aligned global RMSE (mV) | 0.25157 | 0.82458 | **0.24313** |
| Centered/aligned mean-record RMSE (mV) | 0.23374 | 0.81452 | **0.22535** |
| Centered/aligned Pearson r | **0.22492** | 0.07412 | 0.20882 |
| Alignment-boundary fraction (%) | **0.800** | 2.500 | 2.000 |

### Rhythm, QRS, and physical consistency

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| HR MAE to conditioning HR (bpm) | **13.31** | 35.23 | 23.22 |
| HR MAE to paired-real HR (bpm) | **20.30** | 30.49 | 23.83 |
| Within 10 bpm of conditioning HR (%) | **73.31** | 10.11 | 62.67 |
| Within 10 bpm of paired-real HR (%) | **58.90** | 16.20 | 52.60 |
| Beat-count MAE | **3.234** | 4.866 | 3.899 |
| Mean-RR MAE (ms) | **145.75** | 213.80 | 162.30 |
| RR Wasserstein (ms) | **164.37** | 296.32 | 178.99 |
| SDNN MAE (ms) | 78.58 | 216.39 | **78.00** |
| QRS precision | **0.79060** | 0.49538 | 0.72726 |
| QRS recall | 0.76359 | 0.58418 | **0.78255** |
| QRS F1 | **0.77686** | 0.53613 | 0.75389 |
| Matched-QRS timing MAE (ms) | **41.47** | 50.61 | 43.23 |
| Generated QRS detection coverage | **1.000** | **1.000** | **1.000** |
| Generated frontal-identity RMSE (mV) | **0.00270** | 0.06470 | 0.00467 |
| Paired absolute frontal-identity error (mV) | **0.00245** | 0.06442 | 0.00443 |
| Records exceeding 10 mV (%) | **0.000** | 14.100 | **0.000** |

### ECGDeli morphology

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| PR interval MAE (ms) | **19.043** | 28.076 | 22.786 |
| QRS duration MAE (ms) | **8.464** | 16.276 | 10.579 |
| QT interval MAE (ms) | **20.182** | 48.340 | 20.833 |
| QTc Fridericia MAE (ms) | **21.301** | 47.501 | 22.271 |
| ST at J+60 MAE (mV) | **0.01729** | 0.03998 | 0.01835 |
| P-wave duration MAE (ms) | **10.579** | 11.068 | 11.719 |
| T-wave duration MAE (ms) | **9.440** | 32.959 | 10.091 |
| Generated all-metric coverage (%) | 99.900 | 99.600 | **100.000** |
| Paired all-metric coverage (%) | 99.900 | 99.600 | **100.000** |

Morphology rows are median absolute paired-record errors conditional on
successful delineation; the fixed-denominator coverage rows expose every
failed record. This local canonical ECGDeli protocol is shared across the
three models and is not an author-paper evaluator reconstruction.

### DiffuSETS-clean-CLIP64 seed2026 sensitivity

| Metric | ECGDiff e16 | DiffuSETS clean e200 | SE-Diff v3 e195 |
| --- | ---: | ---: | ---: |
| FID | 60929.458 | **45038.105** | 61986.743 |
| Real contiguous-half FID | 122317.338 | 122317.338 | 122317.338 |
| Manifold precision, k=3 | 0.8640 | **1.0000** | 0.7630 |
| Manifold recall, k=3 | **0.9120** | 0.0080 | 0.8920 |
| Manifold F1, k=3 | **0.8874** | 0.0159 | 0.8225 |
| ECG-text cosine | **0.8145** | 0.6342 | 0.8036 |
| Reference ECG-text cosine | 0.8267 | 0.8267 | 0.8267 |
| rCLIP ratio of means | **0.9852** | 0.7672 | 0.9721 |
| Legacy rFID, generated/real-split | 0.4981 | **0.3682** | 0.5068 |
| ICLR rFID, real-split/sum | 0.6675 | **0.7309** | 0.6637 |

The PTB-XL feature space is strongly shifted: the real contiguous-half
FID is about 122,317. Clean DiffuSETS therefore obtains the smallest raw
FID while its recall is 0.008 and manifold F1 is 0.0159. Do not treat
that FID row as evidence of useful coverage.

### Patient-bootstrap uncertainty

| Model | Raw MAE 95% CI | Raw MSE 95% CI | Aligned MAE 95% CI | HR MAE to real 95% CI | Mean-record QRS F1 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| ECGDiff e16 | [0.14664, 0.15464] | [0.07729, 0.12232] | [0.12029, 0.12666] | [18.60804, 21.89189] | [0.77310, 0.79353] |
| DiffuSETS clean e200 | [0.47025, 0.48185] | [0.71510, 0.77047] | [0.45850, 0.46897] | [29.21811, 31.62718] | [0.52421, 0.53348] |
| SE-Diff v3 e195 | [0.14013, 0.14761] | [0.06955, 0.11469] | [0.11785, 0.12358] | [22.08105, 25.58328] | [0.74492, 0.76616] |

The scorer intervals use 1,000 patient-clustered replicates (seed
20260903). The paired clean-minus-baseline intervals below use 10,000
patient-clustered replicates (seed 20260914). Positive error deltas mean
clean DiffuSETS is worse; negative QRS-F1 deltas mean it is worse.

| Baseline | Raw MAE delta [95% CI] | Aligned MAE delta [95% CI] | HR-to-real MAE delta [95% CI] | Mean-record QRS F1 delta [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| ECGDiff e16 | 0.32575 [0.32056, 0.33083] | 0.34061 [0.33568, 0.34562] | 10.19 [8.26, 12.12] | -0.25440 [-0.26654, -0.24208] |
| SE-Diff v3 e195 | 0.33260 [0.32767, 0.33747] | 0.34326 [0.33835, 0.34817] | 6.66 [4.70, 8.67] | -0.22622 [-0.23731, -0.21509] |

## Cohort and contamination boundary

- MIMIC contains the complete 2,149-record native test panel selected for
  ECGDiff epoch 16. Its 419 patients map one-to-one into the clean DiffuSETS
  test role, and its text embeddings match the clean cache bit-for-bit.
- PTB-XL is the fixed recovered local 1,000-record cohort (991 patients, all
  folds). It is external to the MIMIC training corpus and is not the
  authors' unreleased paper cohort. Author-package conditions are retained;
  41 records lack a finite shared HR target, so condition-HR metrics use 959.
- The clean DiffuSETS VAE and U-Net were trained on its patient-disjoint
  MIMIC training role. CLIP64 was trained on train and selected on validation.
  Test records were used only here for evaluation.

## Locked models and inference

- DiffuSETS suite `diffusets-clean-patient-disjoint-seed2026-v1`: VAE `5ade85ed4ff7bfac0b6f5196785c2d452d4279e458f59e54fc34242cf5b3ea0a`, CLIP64
  `45774181d91b61b72d78bfa1add3a59572680a1fdc5737d31a293f521fcbea64`, U-Net `240dc61fa40d6eaa7db21e29757d368ccf9636d6341d0b36525b8595876747aa`. Generation used
  epoch 200/step 54,400, base seed 20260822, 1,000-step ancestral DDPM,
  epsilon prediction, fixed-small variance, predicted-x0 clipping, and no CFG.
- ECGDiff: epoch 16, step 4,624, one locked saved draw per condition.
- SE-Diff: v3 seed 2026 raw epoch 195, checkpoint
  `acac5f55ca4c12c2427dc150a99be2d3affdeab471c0c054ac9745e816fe7861`, 1,000-step ancestral DDPM and CFG 3.
- DiffuSETS generation used clean Git commit
  `3e29005750b67b16a28ec770bd60ea7f11072838` with a clean worktree and
  PyTorch 2.14.0+cu126 on GPUs 2--9.

## Interpretation limits

- This is a fair same-cohort, same-scorer diagnostic. It does not claim the
  local baseline weights reproduce either paper's hidden evaluation protocol.
- One draw cannot measure within-condition diversity. Manifold recall catches
  distributional coverage failure only through this baseline-trained feature
  space.
- DiffuSETS-clean-CLIP64 is the clean reproduction's own learned evaluator,
  so its learned scores are a bespoke sensitivity analysis. Waveform, rhythm,
  QRS, physical-consistency, and ECGDeli rows are the primary independent
  evidence.
- DiffuSETS decoder outputs were explicitly converted from
  `I,II,III,aVR,aVF,aVL,V1--V6` to canonical
  `I,II,III,aVR,aVL,aVF,V1--V6`. The clean evaluator converts them back
  before encoding. The prior released-output aVF/aVL ambiguity is therefore
  absent from this comparison.

## Machine-readable artifacts

- Evaluation root: `/home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp/paper_repro/evaluation/clean_seed2026_e200_base20260822_v1`
- Comparison JSON: `/home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp/paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/comparison.json`

### MIMIC source hashes

| Model | Waveforms SHA-256 | Model-agnostic report SHA-256 | Morphology report SHA-256 | Clean-CLIP64 report SHA-256 |
| --- | --- | --- | --- | --- |
| ECGDiff e16 | `93a437ea8245ae7dda204106167b7cd5c3135ecce60b6aeb9033f3cf58c124af` | `4ec08b75e6efd213092be1ee17ab792c152ba210e6c62bedd07dd0eba4dee89b` | `d3a07c56dc62b9c3f2431a9a362428b326da5b90991964a24047a6cf474c7c97` | `25d5607f0481411681f617675ab792bf6eb1b7df74d79779f6f0ba80ad0399f9` |
| DiffuSETS clean e200 | `362e2a8f307e6034c23b20fb3af06f7bc2daa602af40ad41f7a33ac018439a9f` | `c4e77099c7cb686bf2155a0c440a819265ab60729396fa120cb8acd319658240` | `a53e63c7e50be4f407c10eaede700ee5b0a5d26a3ddb14b815205a31b3a66681` | `470477236c3d2d0c7aa1b9f33746913f8983e8ce4716ed0ee66fe5a6f19cc137` |
| SE-Diff v3 e195 | `1c37209d0b883af1b932f7cfa55ff31cfe9a618f3a626aafaff79e06850ddfea` | `0f2a83123383ca4e314ee18a95d77d81c5d69b49c408489f73b137befd7d6618` | `838687a69b74b4128e338d4f0bf65cd17ad5d1b44b226a186926790561663f04` | `f7f027eb88d1b7b9d694fb106597c0bd4878e2b6a24c593a85806465925481ee` |

### PTBXL source hashes

| Model | Waveforms SHA-256 | Model-agnostic report SHA-256 | Morphology report SHA-256 | Clean-CLIP64 report SHA-256 |
| --- | --- | --- | --- | --- |
| ECGDiff e16 | `c0eba0819aa787b7446efed80d1176ab1c6e987a7076699ba0dad10f46b1ca37` | `b5bb9de0bd7e48e954b7f9dffe2dccda08e4a723ce4c44ac471ec3d459a29966` | `002659b3429616c539ae282530e759c0fdb8119af19620a2083b9add956db2f6` | `55e19677db1456317fb98388353ab200b0e2242044d3fdc2f661e289f8c667c5` |
| DiffuSETS clean e200 | `f3c0e76c688f6de29e121a49fb0c9783ae9a80fb38b1926c981cfab8b556a672` | `77e10bf0a8cbde3d6f7c1d8ca150efca820db7756895524abfc52960229af957` | `d3f7937d2aee36dc9addcb07552fe921d197e2e0dd43a1fc09f456eb6f149f7b` | `8b93daf1f3a81052c97ba347bf638722cf07fbecb4f7f8c9424ffdcab91dc19f` |
| SE-Diff v3 e195 | `5500575e07ede3d76dfc8e88a04b3052a1a361998c6da987a1f1551fa3cf0da2` | `ecb978fe5b4bf37261f8f7bfe647e00490520d157456e161ec2354d1e97df154` | `f9d7d302bf39a721f089e84e31b9b2997831c034b31391a725667c17bc829096` | `b3294cb21f78c1e8263bcad1378d4296ffd8fe2b990b2450e881a189ff3e0c5d` |

## Verification command

```bash
cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp
/home/nvelingker/.conda/envs/ecgdiff/bin/python -m \
  paper_repro.summarize_evaluation \
  --evaluation-root \
  paper_repro/evaluation/clean_seed2026_e200_base20260822_v1 \
  --output-json \
  paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/comparison.json \
  --output-markdown \
  paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/comparison.md \
  --overwrite
```

The command re-hashes every waveform, per-record table, and morphology
artifact; checks all cohort, checkpoint, condition, reference, and evaluator
bindings; and recomputes the paired patient bootstrap before writing output.
