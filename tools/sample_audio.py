#!/usr/bin/env python3
import random
from pathlib import Path
import soundfile as sf
from soundfile import SoundFile
import numpy as np
from tqdm import tqdm
import shutil

# ------------ CONFIG ------------
IN_DIR = Path(
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/separated/podcast_train/00000-of-01432"
)
OUT_DIR = Path("data_sample_audio/podcast_train")
SEG_DUR = 20.0  # 固定抽出長（秒）
N_SAMPLES = 50  # 必ず出力するサンプル数
RANDOM_SEED = 42
CLEAN_OUT = True
DRY_RUN = False


# ------------ HELPERS ------------
def get_duration_sec(wav_path: Path) -> float:
    """WAVファイルの再生時間を秒で取得"""
    with SoundFile(str(wav_path)) as f:
        return f.frames / f.samplerate


# ------------ MAIN ------------
def main():
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {IN_DIR}")

    random.seed(RANDOM_SEED)
    if CLEAN_OUT and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wavs = sorted(IN_DIR.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found in {IN_DIR}")

    print(f"Found {len(wavs)} wavs under {IN_DIR}")

    # 20秒以上の音声のみ対象
    eligible = []
    for p in wavs:
        try:
            dur = get_duration_sec(p)
        except Exception:
            continue
        if dur >= SEG_DUR:
            eligible.append((p, dur))

    if not eligible:
        raise RuntimeError(f"No files longer than {SEG_DUR}s in {IN_DIR}")

    random.shuffle(eligible)
    written = 0
    idx = 0
    pbar = tqdm(total=N_SAMPLES, desc="Sampling podcast_train")

    while written < N_SAMPLES and idx < len(eligible):
        wav_path, total_dur = eligible[idx]
        idx += 1
        try:
            y, sr = sf.read(wav_path)
        except Exception:
            continue

        start = random.uniform(0, total_dur - SEG_DUR)
        end = start + SEG_DUR
        seg = y[int(start * sr) : int(end * sr)]

        out_path = OUT_DIR / f"{written:03d}_{wav_path.stem}_{start:.1f}_{end:.1f}.wav"
        if not DRY_RUN:
            sf.write(out_path, seg, sr)
        written += 1
        pbar.update(1)

    pbar.close()
    print(f"✅ Done: {written} samples written to {OUT_DIR}")
    if written < N_SAMPLES:
        print(f"⚠️ Only {written}/{N_SAMPLES} samples created (not enough long files).")


if __name__ == "__main__":
    main()
