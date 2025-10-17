#!/usr/bin/env python3
import torch
import torchaudio
import soundfile as sf
from typing import List, Tuple
import numpy as np

# --- Silero VAD モデル読み込み ---
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)
(get_speech_timestamps, _, _, _, _) = utils


# --- メイン ---
if __name__ == "__main__":
    wav_path = "data_sample_audio/callhome/000_1003_877.5_897.5.wav"
    audio, sr = sf.read(wav_path)

    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(
            "このスクリプトはステレオ音声（2チャンネル）を前提としています。"
        )

    duration_sec = len(audio) / sr
    scale = 60.0 / duration_sec  # 1分あたり換算

    channel_0 = audio[:, 0]
    channel_1 = audio[:, 1]

    # --- 各チャンネルのIPUとPause ---
    ipus_0, ipu_c0, ipu_dur0, pause_c0, pause_dur0 = extract_ipu_and_pause(
        channel_0, sr
    )
    ipus_1, ipu_c1, ipu_dur1, pause_c1, pause_dur1 = extract_ipu_and_pause(
        channel_1, sr
    )

    # --- Gap / Overlap（両チャンネル間） ---
    gap_c, gap_dur, overlap_c, overlap_dur = compute_gap_overlap(ipus_0, ipus_1)

    # --- 結果出力 ---
    print(f"⏱ Duration: {duration_sec:.2f} sec\n")

    print("🗣️ Channel 0")
    print(f" - IPU count: {ipu_c0} ({ipu_c0 * scale:.2f} / min)")
    print(f" - IPU duration: {ipu_dur0:.2f} sec ({ipu_dur0 * scale:.2f} / min)")
    print(f" - Pause count: {pause_c0} ({pause_c0 * scale:.2f} / min)")
    print(f" - Pause duration: {pause_dur0:.2f} sec ({pause_dur0 * scale:.2f} / min)\n")

    print("🗣️ Channel 1")
    print(f" - IPU count: {ipu_c1} ({ipu_c1 * scale:.2f} / min)")
    print(f" - IPU duration: {ipu_dur1:.2f} sec ({ipu_dur1 * scale:.2f} / min)")
    print(f" - Pause count: {pause_c1} ({pause_c1 * scale:.2f} / min)")
    print(f" - Pause duration: {pause_dur1:.2f} sec ({pause_dur1 * scale:.2f} / min)\n")

    print("🎯 Cross-channel")
    print(f" - Gap count: {gap_c} ({gap_c * scale:.2f} / min)")
    print(f" - Gap duration: {gap_dur:.2f} sec ({gap_dur * scale:.2f} / min)")
    print(f" - Overlap count: {overlap_c} ({overlap_c * scale:.2f} / min)")
    print(
        f" - Overlap duration: {overlap_dur:.2f} sec ({overlap_dur * scale:.2f} / min)"
    )
