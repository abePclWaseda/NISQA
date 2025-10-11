#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -N 0162_nisqa_20s_50

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
DATA_DIR="/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/separated/podcast_test/00000-of-00001/cuts.000000"
OUT_DIR="data_real/podcast_test_20s_head50"
LOG_DIR="logs"
TMP_DIR="tmp_podcast_test_20s_head50_${PBS_JOBID:-local}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$TMP_DIR"

LOG_FILE="${LOG_DIR}/0162_nisqa_20s_head50.${PBS_JOBID:-local}.log"

# ===== Tools detection =====
TRIM_TOOL=""
DUR_TOOL=""
if command -v sox >/dev/null 2>&1; then
  TRIM_TOOL="sox"
  DUR_TOOL="soxi"
elif command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  TRIM_TOOL="ffmpeg"
  DUR_TOOL="ffprobe"
else
  echo "ERROR: sox/soxi もしくは ffmpeg/ffprobe のいずれかが必要です。" >&2
  exit 1
fi

echo "Using TRIM_TOOL=$TRIM_TOOL  DUR_TOOL=$DUR_TOOL"

# ===== Collect up to 50 >=20s files and trim to first 20s =====
TARGET_SECONDS=20
MAX_SAMPLES=50
count=0

# WAV/FLAC/MP3 などを一応拾う（必要に応じて拡張子は調整）
mapfile -t ALL_FILES < <(find "$DATA_DIR" -type f \( -iname "*.wav" -o -iname "*.flac" -o -iname "*.mp3" -o -iname "*.m4a" \) | sort)

if [ ${#ALL_FILES[@]} -eq 0 ]; then
  echo "ERROR: ${DATA_DIR} に音声ファイルが見つかりません。" >&2
  exit 1
fi

echo "Found ${#ALL_FILES[@]} files under ${DATA_DIR}"

for src in "${ALL_FILES[@]}"; do
  # 20秒未満はスキップ
  dur_ok=0
  if [ "$DUR_TOOL" = "soxi" ]; then
    dur=$(soxi -D "$src" || echo 0)
    # soxi が失敗したら0扱い
    awk "BEGIN {exit !($dur >= $TARGET_SECONDS)}" && dur_ok=1 || dur_ok=0
  else
    # ffprobe returns seconds with decimals
    dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$src" 2>/dev/null || echo 0)
    awk "BEGIN {exit !($dur >= $TARGET_SECONDS)}" && dur_ok=1 || dur_ok=0
  fi

  if [ "$dur_ok" -ne 1 ]; then
    continue
  fi

  base="$(basename "$src")"
  # 拡張子を .wav に統一（NISQAはwavが無難）
  dst="${TMP_DIR}/${base%.*}.wav"

  if [ "$TRIM_TOOL" = "sox" ]; then
    # 先頭20秒だけにトリム、フォーマットはsoxに任せる
    sox "$src" "$dst" trim 0 "$TARGET_SECONDS"
  else
    # ffmpeg: re-encodeして20秒へトリム、PCM16LEモノラル/元chでも可だが、無難に元ch維持
    # エンコード負荷を下げたい場合は -acodec copy だとトリム不可ケースがあるため PCM へ
    ffmpeg -nostdin -y -i "$src" -t "$TARGET_SECONDS" -c:a pcm_s16le "$dst" >/dev/null 2>&1
  fi

  # 念のため出力ができたか確認
  if [ ! -s "$dst" ]; then
    echo "WARN: Failed to create trimmed file for $src, skipping." >&2
    continue
  fi

  count=$((count + 1))
  echo "Prepared [$count/$MAX_SAMPLES]: $dst"

  if [ "$count" -ge "$MAX_SAMPLES" ]; then
    break
  fi
done

if [ "$count" -eq 0 ]; then
  echo "ERROR: 20秒以上の音声が見つからず、評価対象が0件でした。" >&2
  exit 2
fi

echo "Collected $count trimmed files (20s each) under $TMP_DIR"

# ===== Run NISQA on the trimmed subset =====
python run_predict.py \
  --mode predict_dir \
  --pretrained_model weights/nisqa.tar \
  --data_dir "$TMP_DIR" \
  --num_workers 0 \
  --bs 10 \
  --ms_max_segments 10000 \
  --output_dir "$OUT_DIR" 2>&1 | tee "$LOG_FILE"

echo "Done. Results in: $OUT_DIR"
echo "Log: $LOG_FILE"
