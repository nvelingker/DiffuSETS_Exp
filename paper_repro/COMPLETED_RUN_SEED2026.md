# Completed clean DiffuSETS run: seed 2026

This is the frozen record for the patient-disjoint production run completed on
2026-09-12. Training used clean Git commit
`d5bb97448064b53eef7519e82759bed161255b3a` on branch
`codex/patient-disjoint-fsdp2`, PyTorch `2.14.0+cu126`, FP32 FSDP2, and GPUs
2--9.

All new inference and evaluation must use the complete checkpoint set below.
Do not use `prerequisites/vae_model.pth`, `prerequisites/clip_model.pth`,
`prerequisites/unet_all.pth`, or generations made with them except for an
explicitly labeled historical audit.

## Launch

```bash
cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
  scripts/train_patient_disjoint_fsdp2.sh \
  config/patient_disjoint_fsdp2.json
```

The launcher executes manifest verification, waveform and repaired-RR
preparation, deterministic posterior-noise creation, contamination audit, VAE
training, latent encoding, CLIP64 training, and diffusion training. FSDP2
resume checkpoints include distributed model/optimizer state and per-rank RNG
state.

## Cohort and prepared data

Subjects use the ECGDiff/SE-Diff split source
`../SE-Diff/data/mimic_iv_ecg_repo1040/manifest.csv` and the first 12 hex digits
of `md5(f"{subject_id}-2026")`: 70% train, 10% validation, and 20% test.

| role | records | patients |
| --- | ---: | ---: |
| train | 556,309 | 111,791 |
| validation | 79,103 | 15,890 |
| test | 158,956 | 31,901 |

Every pairwise patient overlap is zero. The VAE, CLIP64, and diffusion U-Net
receive gradient updates from train only. CLIP selection uses validation only;
VAE selection uses no held-out role; diffusion selection preserves the
released training-loss rule. Test is never used for fitting or selection.

| artifact | relative path | SHA-256 |
| --- | --- | --- |
| configuration | `config/patient_disjoint_fsdp2.json` | `b4f2529a28aa947b050a5d6f8e5c8d4271a90ff13169445b13e8b734adcf8827` |
| manifest | `paper_repro/artifacts/patient_disjoint_v1/manifest.parquet` | `a51858be5dd513dd78cf34dfd2cf58b1f166301404ef7a0da4740b39bfa103bc` |
| waveforms | `paper_repro/artifacts/patient_disjoint_v1/waveforms_float32.npy` | `760475a8a73bd8e790165cad3806cb7f596169e167d7f12d6edaf7532bf71038` |
| repaired HR | `paper_repro/artifacts/patient_disjoint_v1/heart_rate_float32.npy` | `7dbebb9c3d00314fa476ae00ae5237fb8fb2840d621120c5ad341dc9f366a4a7` |
| posterior noise | `paper_repro/artifacts/patient_disjoint_v1/posterior_noise_float32.npy` | `29b054333f95190e02e6e73f371d39ba5bc42c4e3448d74ff9133444c2222c63` |
| latents | `paper_repro/artifacts/patient_disjoint_v1/latents_float32.npy` | `ccd1bb70ec720d296f9f7ba52480ac72f911c9fabeb117906d22e2eb769fa99d` |

Waveforms retain released lead order `I, II, III, aVR, aVF, aVL, V1--V6`,
replace NaNs, and use SciPy FFT resampling from 5,000 to 1,024 samples. Text is
the frozen 1,536-dimensional ada-002 cache; sex and age retain the released
conditions.

The RR repair accepts finite metadata RR values in `[300,1500]` milliseconds
and computes `60000 / RR`; other rows use raw-waveform WFDB XQRS on the first
released-order lead with at least two peaks. This produced 791,163 metadata
rates and 3,205 XQRS rates. It fixes both the released `/1000` unit error and
the undefined `rr_intervals` fallback variable.

## Training contract and outcomes

| stage | parameters | selected outcome |
| --- | --- | --- |
| VAE | seed 2026; 10 epochs; global batch 256; AdamW `1e-4`; OneCycleLR `max_lr=2e-4`; released reconstruction sum-MSE/batch plus Normal KL weighted by `epoch/10`; unshuffled | epoch 10 / file `ep9`, step 21,740; loss 46.427169, reconstruction 38.625026, KL 7.802143 |
| latent encoding | frozen train-only VAE; NumPy PCG64 posterior noise seed 2026 | all 794,368 rows complete; latent SHA above |
| CLIP64 | seed 2026; 10 epochs; embed 64; rank-local contrastive batch 256; global optimizer batch 16,384; AdamW `1e-3`, weight decay `1e-3`; shuffled; validation-only selection | best epoch 2, step 68; validation cosine 0.840531 |
| diffusion | seed 2026; 200 epochs; global batch 2,048; AdamW `1e-4`; cosine LR to `1e-5`; 1,000 DDPM steps; beta `0.00085--0.012`; kernel 7, seven levels; timesteps 1--998; unshuffled | best/final epoch 200, step 54,400; training loss 47.314717 |

The VAE's 10 epochs come from the released training script; the paper does not
state the VAE epoch count. Its criterion was intentionally unchanged. Measured
stage times were about 1 h 6 min for VAE, 12 min 32 s for CLIP, and 4 h 54 min
for diffusion. The resumed end-to-end production launch ran from approximately
13:12 through 19:31 EDT.

## Active checkpoint suite

| component | relative path | SHA-256 |
| --- | --- | --- |
| VAE | `paper_repro/runs/patient_disjoint_seed2026/vae/portable/VAE_model_ep9.pth` | `5ade85ed4ff7bfac0b6f5196785c2d452d4279e458f59e54fc34242cf5b3ea0a` |
| CLIP64 | `paper_repro/runs/patient_disjoint_seed2026/clip/portable/clip_best.pth` | `45774181d91b61b72d78bfa1add3a59572680a1fdc5737d31a293f521fcbea64` |
| diffusion U-Net | `paper_repro/runs/patient_disjoint_seed2026/diffusion/portable/unet_best.pth` | `240dc61fa40d6eaa7db21e29757d368ccf9636d6341d0b36525b8595876747aa` |

Portable files contain a wrapper with `model` and `provenance`. Load them with
the checked loader:

```python
from pathlib import Path

from clip.clip_model import CLIP
from paper_repro.checkpoint import load_portable_payload
from unet.unet_conditional import ECGconditional
from vae.vae_model import VAE_Decoder

run = Path("paper_repro/runs/patient_disjoint_seed2026")
vae_state, vae_provenance = load_portable_payload(
    run / "vae/portable/VAE_model_ep9.pth"
)
clip_state, clip_provenance = load_portable_payload(
    run / "clip/portable/clip_best.pth"
)
unet_state, unet_provenance = load_portable_payload(
    run / "diffusion/portable/unet_best.pth"
)

decoder = VAE_Decoder()
decoder.load_state_dict(vae_state["decoder"], strict=True)
clip = CLIP(embed_dim=64)
clip.load_state_dict(clip_state, strict=True)
unet = ECGconditional(
    1000, kernel_size=7, num_levels=7, n_channels=4, text_embed_dim=1536
)
unet.load_state_dict(unet_state, strict=True)
```

Check the checkpoint hashes before loading. Also require the same config and
manifest hashes in all three provenance objects, the VAE hash in the CLIP
provenance and latent summary, and the latent hash in the diffusion provenance.

The selected suite has not yet produced a registered clean MIMIC or PTB-XL
generation panel. Existing `exp/batch/*` outputs and tables use released
weights and remain historical. Generate a fresh artifact with the clean U-Net,
decode it with the clean VAE, and use the clean CLIP64 for learned metrics.
