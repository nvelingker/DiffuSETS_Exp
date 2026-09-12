#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="/home/nvelingker/.conda/envs/ecgdiff/bin/python"
smoke_root="${1:-paper_repro/smoke/fsdp2_two_rank}"
visible_devices="${CUDA_VISIBLE_DEVICES:-8,9}"
rm -rf "$smoke_root"
smoke_count="${DIFFUSETS_SMOKE_COUNT:-64}"
config_path="$($python_bin -m paper_repro.make_smoke_fixture --output-root "$smoke_root" --count "$smoke_count" | head -n 1)"
export CUDA_VISIBLE_DEVICES="$visible_devices"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
IFS=',' read -r -a gpu_ids <<< "$visible_devices"
torchrun=("$python_bin" -m torch.distributed.run --standalone --nproc-per-node "${#gpu_ids[@]}")

"${torchrun[@]}" -m paper_repro.train vae --config "$config_path" --max-steps 1 --max-epochs 1 --num-workers 0

"${torchrun[@]}" -m paper_repro.train encode --config "$config_path" --num-workers 0
"${torchrun[@]}" -m paper_repro.train clip --config "$config_path" --max-steps 1 --max-epochs 1 --num-workers 0
"${torchrun[@]}" -m paper_repro.train diffusion --config "$config_path" --max-steps 1 --max-epochs 1 --num-workers 0
echo "FSDP2 smoke completed: $smoke_root"
