#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -N 0162_nisqa

set -euxo pipefail

echo "JOB_ID : ${PBS_JOBID:-unknown}"
echo "WORKDIR: ${PBS_O_WORKDIR:-$PWD}"
cd "${PBS_O_WORKDIR:-$PWD}"

# ===== Environment hygiene =====
# Avoid accidental oneAPI/MKL from module env
module purge || true
unset LD_PRELOAD || true

# Limit threads for stability/reproducibility
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
DATA_DIR="/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/separated/podcast_test/00000-of-00001/cuts.000000"
OUT_DIR="data/ground_truth"
LOG_DIR="logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

LOG_FILE="${LOG_DIR}/0162_nisqa.${PBS_JOBID:-local}.log"

# ===== Run =====
python run_predict.py \
  --mode predict_dir \
  --pretrained_model weights/nisqa.tar \
  --data_dir "$DATA_DIR" \
  --num_workers 0 \
  --bs 10 \
  --output_dir "$OUT_DIR" 2>&1 | tee "$LOG_FILE"
