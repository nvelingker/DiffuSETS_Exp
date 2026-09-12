# Patient-disjoint DiffuSETS reproduction

This directory is the clean training path for DiffuSETS. It retains the
released VAE, CLIP64, conditional U-Net, objectives, schedules, batch sizes,
and record preprocessing while replacing the released all-record package with
the same deterministic patient roles used by the local ECGDiff and SE-Diff
experiments. Released learned weights and classifier splits are never loaded.

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

Diffusion remains 200 epochs, global batch 2,048, AdamW at `1e-4`, the released
CosineAnnealingLR, 1,000 diffusion steps, linear beta range 0.00085--0.012,
kernel size 7, seven levels, uniform training timesteps 1--998, and noise
sum-MSE divided by batch size. VAE and diffusion preserve the released
unshuffled manifest order; CLIP preserves released shuffling.

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

Run the complete production dependency chain on GPUs 2--9:

```bash
cd /home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 \
  scripts/train_patient_disjoint_fsdp2.sh \
  config/patient_disjoint_fsdp2.json
```

The script is resumable and executes manifest verification, waveform/RR
preparation, deterministic posterior-noise creation, artifact audit, VAE
training, latent encoding, CLIP64 training, and diffusion training in that
order. It stops on any failed dependency or incomplete cache.

Run the end-to-end two-rank fixture with:

```bash
CUDA_VISIBLE_DEVICES=6,7 \
  scripts/smoke_fsdp2.sh paper_repro/smoke/fsdp2_two_rank
```

CPU/static tests use:

```bash
/home/nvelingker/.conda/envs/ecgdiff/bin/python -m pytest -q tests
```
