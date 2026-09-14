#!/usr/bin/env bash
set -euo pipefail

DIFFUSETS_REPO="/home/nvelingker/arpa-h/diffusion/DiffuSETS_Exp"
SEDIFF_REPO="/home/nvelingker/arpa-h/diffusion/SE-Diff"
PYTHON_BIN="/home/nvelingker/.conda/envs/ecgdiff/bin/python"
EVALUATION_ROOT="${1:-${DIFFUSETS_REPO}/paper_repro/evaluation/clean_seed2026_e200_base20260822_v1}"
CLIP_DEVICE="${CLIP_DEVICE:-cpu}"
PTBXL_TEXT="${SEDIFF_REPO}/data/mimic_iv_ecg_reconstruction_v2/inference/diffusion_iclr_camera_ready_best_suite/inputs/diffusets_ptbxl_saved_text_embeddings_float32.npy"
PTBXL_TEXT_SHA256="6b71e223372c2725c377ec992320df30c136acb4abe50775f84ea987fd0c0019"

inference_dir() {
  case "$1/$2" in
    mimic/diffusets)
      echo "${EVALUATION_ROOT}/mimic"
      ;;
    mimic/ecgdiff)
      echo "${SEDIFF_REPO}/data/mimic_iv_ecg_reconstruction_v3/evaluation/seed_2026_best_e0195_mimic_intersection2149_epoch16_step4624_v1/models/ecgdiff"
      ;;
    mimic/sediff)
      echo "${SEDIFF_REPO}/data/mimic_iv_ecg_reconstruction_v3/evaluation/seed_2026_best_e0195_mimic_intersection2149_epoch16_step4624_v1/models/sediff"
      ;;
    ptbxl/diffusets)
      echo "${EVALUATION_ROOT}/ptbxl"
      ;;
    ptbxl/ecgdiff)
      echo "${SEDIFF_REPO}/data/mimic_iv_ecg_reconstruction_v3/evaluation/epoch16_step4624_ptbxl_local1000_view_v1"
      ;;
    ptbxl/sediff)
      echo "${SEDIFF_REPO}/data/mimic_iv_ecg_reconstruction_v3/exploratory/seed_2026_best_e0195_ptbxl_20260901_v1/test/ptbxl_local1000_1draw"
      ;;
    *)
      return 1
      ;;
  esac
}

cd "${DIFFUSETS_REPO}"
"${PYTHON_BIN}" \
  /home/nvelingker/arpa-h/.agent/skills/diffusets-reproduction/scripts/verify_clean_suite.py

for dataset in mimic ptbxl; do
  for model in diffusets ecgdiff sediff; do
    input="$(inference_dir "${dataset}" "${model}")"
    score_root="${EVALUATION_ROOT}/scores/${dataset}/${model}"
    mkdir -p "${score_root}"

    cd "${SEDIFF_REPO}"
    "${PYTHON_BIN}" -m paper_repro.v2.evaluate \
      --inference-dir "${input}" \
      --output-dir "${score_root}/model_agnostic" \
      --skip-ecgdeli \
      --qrs-workers 8 \
      --bootstrap-replicates 1000 \
      --bootstrap-seed 20260903

    "${PYTHON_BIN}" -m paper_repro.v3.compute_paper_morphology \
      --inference-dir "${input}" \
      --output-dir "${score_root}/paper_morphology_v3" \
      --workers 8 \
      --chunksize 1 \
      --maxtasks-per-child 50 \
      --no-progress

    clip_args=(
      --inference-dir "${input}"
      --output "${score_root}/diffusets_clean_clip64_seed2026.json"
      --batch-size 256
      --device "${CLIP_DEVICE}"
      --overwrite
    )
    if [[ "${dataset}" == "ptbxl" ]]; then
      clip_args+=(
        --text-embeddings "${PTBXL_TEXT}"
        --text-embeddings-sha256 "${PTBXL_TEXT_SHA256}"
      )
    fi
    cd "${DIFFUSETS_REPO}"
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      "${PYTHON_BIN}" -m paper_repro.score_clip64 "${clip_args[@]}"
  done
done

cd "${DIFFUSETS_REPO}"
"${PYTHON_BIN}" -m paper_repro.summarize_evaluation \
  --evaluation-root "${EVALUATION_ROOT}" \
  --output-json "${EVALUATION_ROOT}/comparison.json" \
  --output-markdown "${EVALUATION_ROOT}/comparison.md" \
  --overwrite
