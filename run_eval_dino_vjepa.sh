#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_eval_dino_vjepa.sh
#
# Evaluate Octo + MPPI on CAST validation for both DINOv2 and V-JEPA-2 encoders.
# Results are saved as JSON files for use in the paper.
#
# Usage:
#   bash scripts/run_eval_dino_vjepa.sh                     # defaults
#   bash scripts/run_eval_dino_vjepa.sh --num_examples 100  # override
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults (override via env or CLI passthrough) ───────────────────────────
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
SEED="${SEED:-42}"
PLANNER_TYPES="${PLANNER_TYPES:-octo_mean}"
MPPI_ITERATIONS="${MPPI_ITERATIONS:-4}"
MPPI_SAMPLES="${MPPI_SAMPLES:-32}"
MPPI_ELITES="${MPPI_ELITES:-4}"
MPPI_TEMPERATURE="${MPPI_TEMPERATURE:-0.8}"
MPPI_MAX_STD="${MPPI_MAX_STD:-0.05}"
MPPI_MIN_STD="${MPPI_MIN_STD:-0.01}"
N_OCTO_SAMPLES="${N_OCTO_SAMPLES:-1}"


# ── Checkpoint paths ─────────────────────────────────────────────────────────
# DINOv2
DINO_WM_CKPT="${DINO_WM_CKPT:-/mnt/weka/zhougrp/octo_wm_cast/dino_cast_wm_20260228_193050/best_model.pt}"
DINO_OCTO_DIR="${DINO_OCTO_DIR:-/mnt/weka/zhougrp/octo_pt_cast/cast_dino_finetune_20260219_202345}"
DINO_OCTO_STEP="${DINO_OCTO_STEP:-2000}"

# V-JEPA-2
VJEPA_WM_CKPT="${VJEPA_WM_CKPT:-/mnt/weka/zhougrp/octo_wm_cast/vjepa_cast_wm_20260228_220505/best_model.pt}"
VJEPA_OCTO_DIR="${VJEPA_OCTO_DIR:-/mnt/weka/zhougrp/octo_pt_cast/cast_vjepa_finetune}"
VJEPA_OCTO_STEP="${VJEPA_OCTO_STEP:-2000}"

# ── Output directory ─────────────────────────────────────────────────────────
OUTDIR="${OUTDIR:-eval_results}"
mkdir -p "${OUTDIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

EVAL_SCRIPT="scripts/eval_octo_wm_plan.py"

# ── Common MPPI flags ────────────────────────────────────────────────────────
COMMON_FLAGS=(
    --num_examples "${NUM_EXAMPLES}"
    --seed "${SEED}"
    --mppi_iterations "${MPPI_ITERATIONS}"
    --mppi_samples "${MPPI_SAMPLES}"
    --mppi_elites "${MPPI_ELITES}"
    --mppi_temperature "${MPPI_TEMPERATURE}"
    --mppi_max_std "${MPPI_MAX_STD}"
    --mppi_min_std "${MPPI_MIN_STD}"
    --n_octo_samples "${N_OCTO_SAMPLES}"
)

echo "══════════════════════════════════════════════════════════════════════"
echo "  CAST Evaluation — DINOv2 + V-JEPA-2"
echo "  ${NUM_EXAMPLES} examples, seed=${SEED}, ${TIMESTAMP}"
echo "══════════════════════════════════════════════════════════════════════"

# ── 1. DINOv2 ────────────────────────────────────────────────────────────────
# DINO_JSON="${OUTDIR}/eval_dino_${TIMESTAMP}.json"
# echo ""
# echo "▶ [1/2] Running DINOv2 evaluation..."
# echo "  WM checkpoint : ${DINO_WM_CKPT}"
# echo "  Octo ckpt dir : ${DINO_OCTO_DIR} (step ${DINO_OCTO_STEP})"
# echo "  Output        : ${DINO_JSON}"
# echo ""

# python "${EVAL_SCRIPT}" \
#     --encoder_type dino \
#     --planner_types ${PLANNER_TYPES} \
#     --wm_checkpoint "${DINO_WM_CKPT}" \
#     --octo_ckpt_dir "${DINO_OCTO_DIR}" \
#     --octo_ckpt_step "${DINO_OCTO_STEP}" \
#     --output_json "${DINO_JSON}" \
#     "${COMMON_FLAGS[@]}" \
#     2>&1 | tee "${OUTDIR}/eval_dino_${TIMESTAMP}.log"

# echo ""
# echo "✓ DINOv2 evaluation complete → ${DINO_JSON}"
# echo ""

# ── 2. V-JEPA-2 ─────────────────────────────────────────────────────────────
VJEPA_JSON="${OUTDIR}/eval_vjepa_${TIMESTAMP}.json"
echo "▶ [2/2] Running V-JEPA-2 evaluation..."
echo "  WM checkpoint : ${VJEPA_WM_CKPT}"
echo "  Octo ckpt dir : ${VJEPA_OCTO_DIR} (step ${VJEPA_OCTO_STEP})"
echo "  Output        : ${VJEPA_JSON}"
echo ""

python "${EVAL_SCRIPT}" \
    --encoder_type vjepa \
    --wm_checkpoint "${VJEPA_WM_CKPT}" \
    --planner_types ${PLANNER_TYPES} \
    --octo_ckpt_dir "${VJEPA_OCTO_DIR}" \
    --octo_ckpt_step "${VJEPA_OCTO_STEP}" \
    --output_json "${VJEPA_JSON}" \
    "${COMMON_FLAGS[@]}" \
    2>&1 | tee "${OUTDIR}/eval_vjepa_${TIMESTAMP}.log"

echo ""
echo "✓ V-JEPA-2 evaluation complete → ${VJEPA_JSON}"

# ── Summary ──────────────────────────────────────────────────────────────────
# echo ""
# echo "══════════════════════════════════════════════════════════════════════"
# echo "  All evaluations complete."
# echo "  DINOv2  : ${DINO_JSON}"
# echo "  V-JEPA-2: ${VJEPA_JSON}"
# echo "══════════════════════════════════════════════════════════════════════"
