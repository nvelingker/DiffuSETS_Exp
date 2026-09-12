#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="/home/nvelingker/.conda/envs/ecgdiff/bin/python"
config_path="${1:-config/patient_disjoint_fsdp2.json}"
visible_devices="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7,8,9}"
IFS=',' read -r -a gpu_ids <<< "$visible_devices"
nproc="${#gpu_ids[@]}"
if [[ "$nproc" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must name at least one GPU" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$visible_devices"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

"$python_bin" - <<'PY'
import torch
assert torch.__version__.startswith("2.14."), torch.__version__
print({"torch": torch.__version__, "cuda": torch.version.cuda, "visible_gpus": torch.cuda.device_count()})
PY

"$python_bin" -m paper_repro.prepare manifest --config "$config_path"
"$python_bin" -m paper_repro.prepare waveforms --config "$config_path" --workers "${DIFFUSETS_PREP_WORKERS:-64}" --chunksize 8 --resume
"$python_bin" -m paper_repro.prepare posterior-noise --config "$config_path"
"$python_bin" -m paper_repro.prepare audit --config "$config_path"

torchrun=("$python_bin" -m torch.distributed.run --standalone --nproc-per-node "$nproc")
"${torchrun[@]}" -m paper_repro.train vae --config "$config_path" --resume
"${torchrun[@]}" -m paper_repro.train encode --config "$config_path"
"${torchrun[@]}" -m paper_repro.train clip --config "$config_path" --resume
"${torchrun[@]}" -m paper_repro.train diffusion --config "$config_path" --resume

