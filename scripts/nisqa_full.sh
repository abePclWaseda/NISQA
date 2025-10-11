#!/bin/bash
#PBS -P gcg51557
#PBS -q rt_HG
#PBS -v RTYPE=rt_HG,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -N 0162_nisqa_full

set -euxo pipefail

echo "JOB_ID : ${PBS_JOBID:-unknown}"
echo "WORKDIR: ${PBS_O_WORKDIR:-$PWD}"
cd "${PBS_O_WORKDIR:-$PWD}"

# ===== Threads =====
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ===== Conda =====
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nisqa

echo "==== which python ===="
which python
python --version

# ===== Paths =====
DATA_DIR="/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/output/moshi_stage3_new_jchat_tabidachi/step_498_fp32/continuation_tabidachi_full/generated_wavs"
OUT_DIR="data_tabidachi/moshi_stage3_new_jchat_tabidachi"
LOG_DIR="logs"

mkdir -p "$OUT_DIR" "$LOG_DIR" 

LOG_FILE="${LOG_DIR}/0162_nisqa_full.${PBS_JOBID:-local}.log"

# ===== Run NISQA on the trimmed subset =====
python run_predict.py \
  --mode predict_dir \
  --pretrained_model weights/nisqa.tar \
  --data_dir "$DATA_DIR" \
  --num_workers 0 \
  --bs 10 \
  --ms_max_segments 10000 \
  --output_dir "$OUT_DIR" 2>&1 | tee "$LOG_FILE"

echo "Done. Results in: $OUT_DIR"
echo "Log: $LOG_FILE"
