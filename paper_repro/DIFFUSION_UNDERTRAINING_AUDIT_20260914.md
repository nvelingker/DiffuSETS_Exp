# DiffuSETS seed-2026 v1 diffusion undertraining audit

## Disposition

The clean seed-2026 v1 diffusion U-Net is quarantined. Its poor full-test
result is real, and the primary cause is a hyperparameter-source error: the run
used the later executable `config/all.json` values instead of the values stated
in the paper Methods. The clean VAE, encoded latents, CLIP64 checkpoint,
patient-disjoint manifest, and repaired heart-rate cache remain valid inputs to
the corrected diffusion-only run.

The corrected run is frozen in
`config/patient_disjoint_fsdp2_paper.json` and launches through
`scripts/train_patient_disjoint_diffusion_paper_fsdp2.sh`. It trains only on
the 556,309-record training role and never uses validation or test records for
gradients or checkpoint selection.

## Hyperparameter conflict

The Patterns 2025 Methods section states a batch size of 512 and learning rate
of `5e-4`. The repository README repeats those values. The later
`config/all.json`, added in the January 2025 major-release commit, instead says
batch 2,048 and learning rate `1e-4`. The paper does not state a diffusion epoch
count, so the corrected recipe retains the released code's 200 epochs,
CosineAnnealingLR, and training-loss checkpoint selection.

| Recipe | global batch | learning rate | clean steps/epoch | 200-epoch updates |
| --- | ---: | ---: | ---: | ---: |
| quarantined v1 / `config/all.json` | 2,048 | `1e-4` | 272 | 54,400 |
| corrected paper Methods | 512 | `5e-4` | 1,087 | 217,400 |
| released all-record package at paper batch (reference) | 512 | `5e-4` | 1,552 | 310,400 |

The v1 learning rate also decayed to `1e-5` by its final update. Its epoch loss
was still falling: 103.7780 at epoch 50, 62.8521 at 100, 51.0026 at 150, and
47.3147 at 200. That trajectory alone suggested incomplete optimization, but
the checkpoint-level diagnostics below establish the failure directly.

## Fixed-noise denoising diagnostic

We evaluated 64 deterministic examples with fixed Gaussian noise at nine DDPM
timesteps. Clean v1 checkpoints used both clean train and patient-disjoint test
latents; the released checkpoint used the authors' first 64 packaged latents.
Train and test results for v1 were nearly identical, which rules out held-out
distribution shift as the explanation.

| checkpoint | t=50 MSE | t=250 MSE | t=500 MSE | t=998 MSE |
| --- | ---: | ---: | ---: | ---: |
| clean v1 epoch 50, test | 0.37575 | 0.19029 | 0.16614 | 0.16290 |
| clean v1 epoch 100, test | 0.29715 | 0.11340 | 0.08797 | 0.08358 |
| clean v1 epoch 150, test | 0.27427 | 0.09100 | 0.06506 | 0.06040 |
| clean v1 epoch 200, test | 0.26622 | 0.08427 | 0.05808 | 0.05331 |
| author-released checkpoint, author package | 0.18451 | 0.03099 | 0.00509 | 0.000117 |

MSE is per latent element. The v1 checkpoint improves monotonically but remains
orders of magnitude behind the released model at high-noise timesteps. At
`t=998`, its predicted-noise correlation is 0.9740 versus 0.99994 for the
released checkpoint.

## End-to-end DDPM latent diagnostic

Starting from the same fixed Gaussian noise and eight fixed author conditions,
the exact 1,000-step sampler produced these terminal latent distributions:

| model | terminal mean | terminal std | min | max | fraction at `abs(z)>=0.999` |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean v1 epoch 200 | -0.0145 | 0.3949 | -1.0000 | 1.0000 | 1.59% |
| author-released | 0.0115 | 0.1466 | -0.5840 | 0.6384 | 0% |

The clean latent cache has standard deviation about 0.1796. The v1 sampler is
therefore producing latents far outside the decoder's training distribution,
which explains the large-amplitude ECGs and poor waveform, rhythm, and
clean-CLIP64 metrics. The released sampler lands close to its training latent
scale.

## Components ruled out

- The retrained VAE reconstructs a deterministic 512-record check panel with
  MAE 0.03368 mV; reconstructed standard deviation is 0.21294 mV versus
  0.22253 mV for real signals. Decoder scaling is not the source of the
  generated amplitude failure.
- The U-Net has no BatchNorm. It uses GroupNorm and LayerNorm, so rank-local
  running statistics cannot explain the FSDP result.
- A two-rank PyTorch 2.14 FSDP2 AdamW update matched an unsharded global-batch
  update to maximum parameter error `7.28e-12`. Installed source confirms FP32
  reduce-scatter uses an averaged reduction.
- The clean and released U-Nets have the same 713 state keys, tensor shapes,
  and 14,177,792 parameters/buffers.
- The first 10,000 clean manifest rows match the authors' lite package exactly
  for subject, report, sex, age, and all 1,536 text-embedding values. Heart
  rates intentionally differ because the requested RR repair replaces the
  released unit/fallback behavior.
- Train and test clean latent distributions match closely (sampled standard
  deviations 0.17944 and 0.18001), so the patient split did not create a latent
  scale shift.
- DDPM betas, timestep sampling, epsilon objective, condition ordering,
  checkpoint loading, and inference stepping match the released code.

## Corrected production command

```bash
cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
  scripts/train_patient_disjoint_diffusion_paper_fsdp2.sh \
  config/patient_disjoint_fsdp2_paper.json
```

This command revalidates the clean inputs and then trains the corrected U-Net
from scratch. It does not reload the quarantined U-Net or any author-released
model weights. Acceptance requires the same fixed-noise and terminal-latent
diagnostics before a new full test evaluation is registered.

For a portable candidate checkpoint, run the reproducible diagnostic with:

```bash
CUDA_VISIBLE_DEVICES=9 \
  /home/nvelingker/.conda/envs/ecgdiff/bin/python \
  -m paper_repro.diagnose_diffusion \
  --config config/patient_disjoint_fsdp2_paper.json \
  --checkpoint paper_repro/runs/patient_disjoint_seed2026_paper/diffusion/portable/unet_200.pth \
  --output paper_repro/runs/patient_disjoint_seed2026_paper/diffusion/diagnostic_e200.json \
  --device cuda:0
```

The tracked machine-readable v1 reference is
`paper_repro/resources/diffusion_undertraining_diagnostics_20260914.json`.
