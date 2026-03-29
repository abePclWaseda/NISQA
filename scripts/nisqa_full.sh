#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
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
DATA_DIR="/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/output/moshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_and_VisualBank_7epochs_1node_exp_textpad1/step_12516_fp32/continuation_llmjp-zoom1_test/generated_wavs"
OUT_DIR="data_llmjp-zoom1_test/tesmoshi-finetuned_init_text_emb_train_ohashi_llmjp-zoom1_and_VisualBank_7epochs_1node_exp_textpad1"
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
  --ms_max_segments 15000 \
  --output_dir "$OUT_DIR" 2>&1 | tee "$LOG_FILE"

echo "Done. Results in: $OUT_DIR"
echo "Log: $LOG_FILE"

# ===== Check MOS scores =====
CSV_FILE="${OUT_DIR}/NISQA_results.csv"
if [ -f "$CSV_FILE" ]; then
    echo "==== Checking MOS scores ====" | tee -a "$LOG_FILE"
    python tools/check_mos.py "$CSV_FILE" | tee -a "$LOG_FILE"
else
    echo "Warning: NISQA_results.csv not found: $CSV_FILE" | tee -a "$LOG_FILE"
fi
