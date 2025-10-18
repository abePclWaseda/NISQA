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


# --- IPUとPause抽出 ---
def extract_ipu_and_pause(
    audio: np.ndarray, sr: int, silence_thresh: float = 0.2, target_sr: int = 16000
) -> Tuple[List[Tuple[float, float]], int, float, int, float]:
    """
    Silero VAD を用いて IPU（発話区間）および Pause（無音区間）を抽出する。
    silence_thresh 未満の無音区間は IPU 結合対象とする。
    """
    # --- リサンプリング ---
    if sr not in [8000, 16000]:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        audio_tensor = transform(audio_tensor)
        waveform = audio_tensor.squeeze(0)
        sr = target_sr
    else:
        waveform = torch.from_numpy(audio).float()

    # --- 音声区間の推定 ---
    speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=sr)
    ipus = [(seg["start"] / sr, seg["end"] / sr) for seg in speech_timestamps]

    # --- 近接IPUの結合 ---
    merged_ipus = []
    for seg in ipus:
        if not merged_ipus:
            merged_ipus.append(seg)
        else:
            prev_s, prev_e = merged_ipus[-1]
            curr_s, curr_e = seg
            if curr_s - prev_e < silence_thresh:
                # ギャップが閾値未満なら結合
                merged_ipus[-1] = (prev_s, curr_e)
            else:
                merged_ipus.append(seg)

    ipus = merged_ipus
    ipu_count = len(ipus)
    ipu_duration_total = sum(e - s for s, e in ipus)

    # --- 無音区間計算 ---
    pause_count = 0
    pause_total = 0.0
    for (s1, e1), (s2, e2) in zip(ipus, ipus[1:]):
        gap = s2 - e1
        if gap >= silence_thresh:
            pause_count += 1
            pause_total += gap

    return ipus, ipu_count, ipu_duration_total, pause_count, pause_total


# --- Gap / Overlap 計算 ---
def compute_gap_overlap(
    ipus_a: List[Tuple[float, float]],
    ipus_b: List[Tuple[float, float]],
) -> Tuple[int, float, int, float]:
    """
    2チャンネルのIPUから、話者が異なる隣接発話間のGapおよびOverlapを計算。
    二本指スイープで、毎回「A区間 vs B区間」の組を一つだけ評価し、
    早く終わる方を進めることで、過不足なく数え上げる。
    """
    i = j = 0
    gap_count = 0
    gap_total = 0.0
    overlap_count = 0
    overlap_total = 0.0

    while i < len(ipus_a) and j < len(ipus_b):
        a_s, a_e = ipus_a[i]
        b_s, b_e = ipus_b[j]

        # 完全に離れている（Aが先に終わる → Aの終わりからBの始まりがGap）
        if a_e <= b_s:
            d = b_s - a_e
            if d > 0:
                gap_count += 1
                gap_total += d
            i += 1
            continue

        # 完全に離れている（Bが先に終わる → Bの終わりからAの始まりがGap）
        if b_e <= a_s:
            d = a_s - b_e
            if d > 0:
                gap_count += 1
                gap_total += d
            j += 1
            continue

        # 重なっている（Overlap）
        start = max(a_s, b_s)
        end = min(a_e, b_e)
        d = end - start
        if d > 0:
            overlap_count += 1
            overlap_total += d

        # 早く終わる側を進める（被覆や部分包含も自然に分割される）
        if a_e <= b_e:
            i += 1
        else:
            j += 1

    # 表示の安定性のため軽く丸める（必要なら外してOK）
    return gap_count, round(gap_total, 3), overlap_count, round(overlap_total, 3)


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
