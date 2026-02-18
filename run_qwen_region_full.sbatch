#!/bin/bash -l
#SBATCH --job-name=qwen_omni_region
#SBATCH --partition=h24gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qwen_omni_region_%j.out
#SBATCH --account=OD-237777
#SBATCH --gres=gpu:1

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

conda activate deepfake

if [[ -n "${SCRIPT_DIR_OVERRIDE:-}" ]]; then
  SCRIPT_DIR="${SCRIPT_DIR_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_qwen_region_full.sbatch" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -n "${SLURM_JOB_SCRIPT:-}" && -f "${SLURM_JOB_SCRIPT}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${SLURM_JOB_SCRIPT}")")" && pwd)"
elif [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi

mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/ALLM/Qwen3-Omni-30B-A3B-Thinking/}"
REGION_CSV="${REGION_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv}"
WAV_ROOT="${WAV_ROOT:-/datasets/work/dss-deepfake-audio/work/data/voc-v4/voc.v4/wav/}"
SPEC_ROOT="${SPEC_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid/}"
INPUT_JSONL="${INPUT_JSONL:-}"
SYSTEM_FILE="${SYSTEM_FILE:-${SCRIPT_DIR}/prompts/region_forensics_system.txt}"
USER_TEMPLATE_FILE="${USER_TEMPLATE_FILE:-${SCRIPT_DIR}/prompts/region_forensics_user.txt}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-${SCRIPT_DIR}/outputs/qwen3_omni}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}"

if [[ "${SHARD_COUNT}" -gt 1 ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}/shard_${SHARD_ID}_of_${SHARD_COUNT}}"
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_BASE_DIR}/qwen_omni_outputs_shard_${SHARD_ID}_of_${SHARD_COUNT}.jsonl}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE_DIR}}"
  OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/qwen_omni_outputs.jsonl}"
fi

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
DTYPE="${DTYPE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
FLASH_ATTN2="${FLASH_ATTN2:-0}"
USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-1}"

EXTRA_ARGS=()
if [[ "${DO_SAMPLE}" == "1" ]]; then
  EXTRA_ARGS+=(--do-sample)
fi
if [[ "${FLASH_ATTN2}" == "1" ]]; then
  EXTRA_ARGS+=(--flash-attn2)
fi
if [[ "${USE_AUDIO_IN_VIDEO}" != "1" ]]; then
  EXTRA_ARGS+=(--no-use-audio-in-video)
fi

python "${SCRIPT_DIR}/qwen_region_artifact_prompt.py" \
  --model-id "${MODEL_ID}" \
  --region-csv "${REGION_CSV}" \
  --wav-root "${WAV_ROOT}" \
  --spec-root "${SPEC_ROOT}" \
  --system-file "${SYSTEM_FILE}" \
  --user-template-file "${USER_TEMPLATE_FILE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --batch-size "${BATCH_SIZE}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --num-shards "${SHARD_COUNT}" \
  --shard-id "${SHARD_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  "${EXTRA_ARGS[@]}" \
  ${INPUT_JSONL:+--input-jsonl "${INPUT_JSONL}"} \
  "$@"
