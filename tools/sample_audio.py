#!/usr/bin/env python3
import random
from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm

# ------------ CONFIG ------------
IN_DIR = Path("/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/CSJ/audio")
OUT_DIR = Path("data_sample_audio/CSJ")
N_SAMPLES = 50  # 抽出するファイル数
SEG_MIN = 10.0  # 最小抽出長（秒）
SEG_MAX = 30.0  # 最大抽出長（秒）
RANDOM_SEED = 42
CLEAN_OUT = True  # 既存ディレクトリ削除
DRY_RUN = False  # Trueで確認のみ

# ------------ MAIN ------------
random.seed(RANDOM_SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 音声ファイル一覧取得
wavs = sorted(IN_DIR.rglob("*.wav"))
if len(wavs) == 0:
    raise FileNotFoundError(f"No wav files found in {IN_DIR}")

print(f"Found {len(wavs)} wavs under {IN_DIR}")

# サンプリング対象をランダム選択
sample_files = random.sample(wavs, min(N_SAMPLES, len(wavs)))

for i, wav_path in enumerate(tqdm(sample_files, desc="Sampling")):
    y, sr = sf.read(wav_path)
    total_dur = len(y) / sr
    if total_dur < SEG_MIN:
        continue

    # 開始位置をランダムに
    seg_dur = random.uniform(SEG_MIN, min(SEG_MAX, total_dur))
    start = random.uniform(0, total_dur - seg_dur)
    end = start + seg_dur

    seg = y[int(start * sr) : int(end * sr)]

    out_path = OUT_DIR / f"{i:03d}_{wav_path.stem}_{start:.1f}_{end:.1f}.wav"

    if not DRY_RUN:
        sf.write(out_path, seg, sr)

print(f"✅ Done: {len(list(OUT_DIR.glob('*.wav')))} samples written to {OUT_DIR}")
