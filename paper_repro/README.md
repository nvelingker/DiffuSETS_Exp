# Patient-disjoint DiffuSETS reproduction

## Diffusion hyperparameter correction (2026-09-14)

The seed-2026 v1 diffusion U-Net is quarantined after the full test evaluation.
It used the January 2025 `config/all.json` values (global batch 2,048 and
learning rate `1e-4`), but the paper Methods and repository README specify
batch 512 and learning rate `5e-4`. The v1 run therefore made 54,400 optimizer
updates, and fixed-noise diagnostics show that it remained severely
undertrained. Do not use its U-Net or its generated samples for new results.

The clean VAE, encoded latents, CLIP64 evaluator, patient split, and repaired-RR
cache remain valid: none used validation/test records for fitting and the VAE
reconstruction audit passed. The corrected diffusion-only run is defined by
`config/patient_disjoint_fsdp2_paper.json` and
`scripts/train_patient_disjoint_diffusion_paper_fsdp2.sh`. It keeps the
released-code 200-epoch schedule because the paper does not state an epoch
count, while taking batch size and learning rate from the paper.

See [DIFFUSION_UNDERTRAINING_AUDIT_20260914.md](DIFFUSION_UNDERTRAINING_AUDIT_20260914.md)
for the measurements and disposition of the v1 checkpoint.

This directory is the clean training path for DiffuSETS. It retains the
released VAE, CLIP64, conditional U-Net, objectives, schedules, and record
preprocessing while replacing the released all-record package with the same
deterministic patient roles used by the local ECGDiff and SE-Diff experiments.
Released learned weights and classifier splits are never loaded for training.

The completed seed-2026 production run, including exact data/checkpoint paths,
hashes, selected epochs, parameters, timings, and portable loading examples, is
recorded in [COMPLETED_RUN_SEED2026.md](COMPLETED_RUN_SEED2026.md). Its VAE,
latent cache, and CLIP64 checkpoint remain the clean dependencies for the
corrected run. Its diffusion U-Net is retained only for audit. Author-released
checkpoints under `prerequisites/` are also historical-audit artifacts only.

## Fixed cohort

`config/patient_disjoint_fsdp2.json` binds every input by SHA-256. The source
of truth is `../SE-Diff/data/mimic_iv_ecg_repo1040/manifest.csv`. A subject is
assigned from the first 12 hexadecimal digits of
`md5(f"{subject_id}-2026")`: values below 0.70 are train, values below 0.80 are
validation, and the rest are test.

The clean manifest also proves row identity against ECGDiff's prepared
DiffuSETS manifest and conditioning cache. After the released 5,663 exclusion
indices and four records with no usable 12-lead waveform are removed, its
frozen counts are:

| role | records | patients |
| --- | ---: | ---: |
| train | 556,309 | 111,791 |
| validation | 79,103 | 15,890 |
| test | 158,956 | 31,901 |
| total | 794,368 | 159,582 |

All three pairwise patient overlaps are zero. The manifest refuses to build if
the source hashes, row order, subject identities, split roles, conditioning
shape, or counts differ.

The waveform cache preserves the raw MIMIC/DiffuSETS channel order
`I, II, III, aVR, aVF, aVL, V1--V6` and uses the released operation exactly:
NaN replacement followed by
`scipy.signal.resample(raw_5000x12, 1024, axis=0)` and float32 storage. Text
uses the existing frozen 1,536-dimensional `text-embedding-ada-002` cache. Sex
and age use the released conditions. These inputs have no locally fitted
parameters.

## RR repair

MIMIC `rr_interval` is in milliseconds. The released loader divided it by
1,000 and then compared the result with 300 and 1,500, so every normal record
entered XQRS. Its other branch referenced the undefined name `rr_intervals`.
The repaired rule is:

1. accept finite RR values from 300 through 1,500 ms;
2. compute `heart_rate = 60000 / rr_interval_ms`;
3. otherwise run WFDB XQRS on the raw leads in released lead order and use the
   first lead with at least two peaks; and
4. fail preparation if no valid estimate exists.

The cache records whether each value came from metadata or XQRS. A failed or
pending row blocks all training; it is never silently removed.

## Contamination boundary

| learned component | training records | selection records | policy |
| --- | --- | --- | --- |
| VAE encoder/decoder | train only | none | retrain from scratch |
| DiffuSETS-CLIP64 | train only | validation only | retrain from scratch |
| conditional diffusion U-Net | train only | training loss, as released | retrain from scratch |

Encoding validation and test waveforms with the frozen train-only VAE is a
transform, not fitting. Test records are never used for gradient updates,
checkpoint selection, or early stopping. Each checkpoint embeds the manifest,
configuration, dependency, and artifact hashes plus explicit exposure roles.

The released downstream classifier artifacts remain quarantined. Their
40k/5k/5k record-key split omits patient identities, and the released loop
selects checkpoints on its test loader. Those files cannot provide an unbiased
classification result and are not dependencies of this pipeline.

## Preserved training contract

The VAE remains 10 epochs, global batch 256, AdamW at `1e-4`, OneCycleLR with
`max_lr=2e-4`, and the unmodified released loss: reconstruction sum-MSE divided
by batch size plus the released Normal KL weighted by epoch `1/10` through
`10/10`. Compatibility checkpoints retain zero-based names `ep6` through
`ep9`; `VAE_model_ep9.pth` feeds latent encoding.

CLIP remains the released 64-dimensional model, local contrastive batch 256,
effective optimizer batch 16,384, AdamW at `1e-3` with weight decay `1e-3`,
and 10 epochs. Its released self-evaluation on its training archive is replaced
with selection on the patient-disjoint validation role. Rank-local contrastive
sets remain size 256; FSDP2 suppresses gradient communication during
microbatch accumulation. Replicated BatchNorm running statistics are averaged
across ranks before validation and checkpointing.

Corrected diffusion remains 200 epochs, AdamW with the released
CosineAnnealingLR, 1,000 diffusion steps, linear beta range 0.00085--0.012,
kernel size 7, seven levels, uniform training timesteps 1--998, and noise
sum-MSE divided by batch size. It uses the paper's global batch 512 and
learning rate `5e-4`. VAE and diffusion preserve the released unshuffled
manifest order; CLIP preserves released shuffling.

FP32 is intentional: the released scripts do not enable mixed precision, and
PyTorch 2.14 FSDP2 parameter-only BF16 casting is incompatible with the
released CLIP BatchNorm FP32 running buffers.

Training samplers shard one deterministic global order without padding: every
training record appears exactly once per epoch, with no repeated records added
to make rank lengths equal. The fixed production counts still give every rank
the same number of collective steps. The uneven final batch is weighted by its
true rank-local record count before FSDP's gradient average.

## FSDP2 implementation

Run with `/home/nvelingker/.conda/envs/ecgdiff/bin/python`, currently PyTorch
2.14.x. `paper_repro/fsdp2.py` uses composable
`torch.distributed.fsdp.fully_shard`, not FSDP1. Semantic child groups are
sharded bottom-up and the root is sharded last. Child groups reshard after
forward; the root retains parameters until backward. The optimizer is always
created after sharding. Microbatch accumulation uses
`set_requires_gradient_sync`, and `set_is_last_backward` marks the final
backward.

The implementation was checked against the installed 2.14 sources:

- `torch/distributed/fsdp/_fully_shard/_fully_shard.py`
- `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py`
- `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`
- `torch/distributed/checkpoint/state_dict.py`

Resume checkpoints use the canonical distributed-checkpoint
`get_state_dict`/`set_state_dict` APIs and include per-rank Python, NumPy, CPU,
and CUDA RNG state. Portable rank-zero full state dictionaries preserve the
released VAE/CLIP/U-Net key layouts. Exact RNG resume requires the same world
size.

## Commands

The completed v1 dependency-chain command below is retained for provenance; do
not relaunch it because its diffusion section contains the conflicting later
JSON settings:

```bash
cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
  scripts/train_patient_disjoint_fsdp2.sh \
  config/patient_disjoint_fsdp2.json
```

That script is resumable and executes manifest verification, waveform/RR
preparation, deterministic posterior-noise creation, artifact audit, VAE
training, latent encoding, CLIP64 training, and diffusion training in that
order. It stops on any failed dependency or incomplete cache.

Run the corrected diffusion stage on GPUs 2--9 with:

```bash
cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
  scripts/train_patient_disjoint_diffusion_paper_fsdp2.sh \
  config/patient_disjoint_fsdp2_paper.json
```

Run the end-to-end two-rank fixture with:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
  scripts/smoke_fsdp2.sh paper_repro/smoke/fsdp2_two_rank
```

CPU/static tests use:

```bash
/home/nvelingker/.conda/envs/ecgdiff/bin/python -m pytest -q tests
```

## Quarantined v1 test inference and learned scoring

`paper_repro/infer.py` currently records the generation path used for the
quarantined v1 evaluation. It rejects any VAE or U-Net other than the v1
hash-bound seed-2026 files,
requires a clean Git checkout, preserves one deterministic random stream per
condition, and uses the released 1,000-step ancestral DDPM defaults. The
decoder emits the raw DiffuSETS lead order with `aVF` before `aVL`; the adapter
swaps those two channels before saving the canonical comparison order
`I, II, III, aVR, aVL, aVF, V1--V6`.

Do not run the commands in this section for a new result. They remain here to
reproduce the failure audit. The checkpoint locks will be replaced only after
the corrected U-Net passes the denoising and terminal-latent acceptance checks.

The MIMIC comparison contains all 2,149 native test records from the selected
ECGDiff e16 run, representing 419 held-out patients. Its records map one-to-one
to the clean DiffuSETS test role, and its 1,536-dimensional text inputs match
the clean cache bit-for-bit. The external PTB-XL comparison retains the fixed
1,000-record/991-patient local cohort and its recovered saved ada-002 inputs.

Run MIMIC on eight GPUs with:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
  /home/nvelingker/.conda/envs/ecgdiff/bin/torchrun \
  --standalone --nproc-per-node=8 -m paper_repro.infer \
  --condition-source clean-mimic \
  --condition-dir ../SE-Diff/data/mimic_iv_ecg_reconstruction_v3/evaluation/seed_2026_best_e0195_mimic_intersection2149_epoch16_step4624_v1/conditions \
  --output-dir paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/mimic \
  --base-seed 20260822 --batch-size-per-rank 512
```

Run the fixed PTB-XL panel with:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
  /home/nvelingker/.conda/envs/ecgdiff/bin/torchrun \
  --standalone --nproc-per-node=8 -m paper_repro.infer \
  --condition-source ptbxl-author-package \
  --condition-dir ../SE-Diff/data/mimic_iv_ecg_reconstruction_v2/evaluation/ptbxl_diffusets_local1000 \
  --text-embeddings ../SE-Diff/data/mimic_iv_ecg_reconstruction_v2/inference/diffusion_iclr_camera_ready_best_suite/inputs/diffusets_ptbxl_saved_text_embeddings_float32.npy \
  --text-embeddings-sha256 6b71e223372c2725c377ec992320df30c136acb4abe50775f84ea987fd0c0019 \
  --output-dir paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/ptbxl \
  --base-seed 20260822 --batch-size-per-rank 512
```

`paper_repro/score_clip64.py` loads only the clean validation-selected CLIP64
checkpoint. It converts canonical saved waveforms back to the DiffuSETS
training lead order inside the evaluator. Its FID, manifold precision/recall,
CLIP, rCLIP, and rFID outputs must be labeled
`DiffuSETS-clean-CLIP64 seed2026`; they remain a bespoke sensitivity analysis,
not a model-independent ranking.

## Full test comparison

The completed comparison uses the entire 2,149-record/419-patient MIMIC panel
and the fixed 1,000-record/991-patient PTB-XL panel. Each clean DiffuSETS
condition uses base seed `20260822` plus its zero-based row position. ECGDiff is
locked to epoch 16/step 4,624 and SE-Diff to v3 seed-2026 raw epoch 195.

Run every model-agnostic, morphology, and clean-CLIP64 scorer and regenerate the
hash-checked reports with:

```bash
cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp
scripts/evaluate_clean_test_comparison.sh
```

The scorer uses 1,000 patient-bootstrap replicates with seed `20260903`; the
aggregator additionally computes 10,000 paired patient-bootstrap replicates
with seed `20260914`. Set `CLIP_DEVICE=cuda:0` to run only the small learned
evaluator on a free GPU; its default is CPU.

The registered human-readable report is
[`COMPLETED_EVALUATION_SEED2026.md`](COMPLETED_EVALUATION_SEED2026.md). The full
machine-readable output and per-record scores live under
`paper_repro/evaluation/clean_seed2026_e200_base20260822_v1/` and remain local
because waveform/evaluation artifacts are gitignored. Re-run
`paper_repro.summarize_evaluation` to verify every checkpoint, condition,
waveform, scorer, and per-record-artifact hash before using a number.
