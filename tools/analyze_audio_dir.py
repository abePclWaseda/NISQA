#!/usr/bin/env python3
import glob
import os
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ========= 設定 =========
DATA_DIR = "data_sample_audio/callhome"  # 集計したいディレクトリ
PATTERN = "**/*.wav"                     # 再帰検索
MERGE_SILENCE_THRESH = 0.2               # 近接IPU結合用（Silero VADの出力マージ用）

# ========= Silero VAD =========
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)
(get_speech_timestamps, _, _, _, _) = utils


# --- IPU抽出（Pauseはここでは返さない） ---
def extract_ipu_and_pause(
    audio: np.ndarray, sr: int, silence_thresh: float = MERGE_SILENCE_THRESH, target_sr: int = 16000
) -> Tuple[List[Tuple[float, float]], int, float]:
    """
    Silero VAD でIPU（発話区間）列を得て、近接IPUを結合。
    返り値: (ipus, ipu_count, ipu_duration_total)
    """
    # リサンプリング
    if sr not in [8000, 16000]:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        audio_tensor = transform(audio_tensor)
        waveform = audio_tensor.squeeze(0)
        sr = target_sr
    else:
        waveform = torch.from_numpy(audio).float()

    # VAD
    speech_timestamps = get_speech_timestamps(waveform, model, sampling_rate=sr)
    ipus = [(seg["start"] / sr, seg["end"] / sr) for seg in speech_timestamps]

    # 近接IPUの結合
    merged_ipus: List[Tuple[float, float]] = []
    for seg in ipus:
        if not merged_ipus:
            merged_ipus.append(seg)
        else:
            prev_s, prev_e = merged_ipus[-1]
            curr_s, curr_e = seg
            if curr_s - prev_e < silence_thresh:
                merged_ipus[-1] = (prev_s, curr_e)
            else:
                merged_ipus.append(seg)

    ipus = merged_ipus
    ipu_count = len(ipus)
    ipu_duration_total = sum(e - s for s, e in ipus)
    return ipus, ipu_count, ipu_duration_total


# --- Mutual Pause（相手が完全沈黙のときだけ） ---
def compute_mutual_pause(
    ipus_self: List[Tuple[float, float]],
    ipus_other: List[Tuple[float, float]],
) -> Tuple[int, float, List[Tuple[float, float]]]:
    """
    同一話者の連続IPU間ギャップのうち、相手のIPUと一点でも重ならない区間のみ Pause として採用。
    """
    pause_count = 0
    pause_total = 0.0
    pause_segments: List[Tuple[float, float]] = []

    j = 0
    n_other = len(ipus_other)

    for (s1, e1), (s2, e2) in zip(ipus_self, ipus_self[1:]):
        gap_s, gap_e = e1, s2
        if gap_e <= gap_s:
            continue

        # other を gap_s 以降まで進める
        while j < n_other and ipus_other[j][1] <= gap_s:
            j += 1

        # 重なりがあれば不採用
        has_overlap = False
        k = j
        while k < n_other and ipus_other[k][0] < gap_e:
            other_s, other_e = ipus_other[k]
            if other_e > gap_s:
                has_overlap = True
                break
            k += 1

        if not has_overlap:
            d = gap_e - gap_s
            if d > 0:
                pause_count += 1
                pause_total += d
                pause_segments.append((gap_s, gap_e))

    return pause_count, pause_total, pause_segments


# --- Gap / Overlap（二本指スイープ） ---
def compute_gap_overlap(
    ipus_a: List[Tuple[float, float]],
    ipus_b: List[Tuple[float, float]],
) -> Tuple[int, float, int, float]:
    i = j = 0
    gap_count = 0
    gap_total = 0.0
    overlap_count = 0
    overlap_total = 0.0

    while i < len(ipus_a) and j < len(ipus_b):
        a_s, a_e = ipus_a[i]
        b_s, b_e = ipus_b[j]

        # 完全に離れている（Aが先）
        if a_e <= b_s:
            d = b_s - a_e
            if d > 0:
                gap_count += 1
                gap_total += d
            i += 1
            continue

        # 完全に離れている（Bが先）
        if b_e <= a_s:
            d = a_s - b_e
            if d > 0:
                gap_count += 1
                gap_total += d
            j += 1
            continue

        # 重なり
        start = max(a_s, b_s)
        end = min(a_e, b_e)
        d = end - start
        if d > 0:
            overlap_count += 1
            overlap_total += d

        # 早く終わる側を進める
        if a_e <= b_e:
            i += 1
        else:
            j += 1

    return gap_count, gap_total, overlap_count, overlap_total


def per_minute(count_or_seconds: float, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return count_or_seconds / (duration_sec / 60.0)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN), recursive=True))
    if not files:
        print(f"No wav files found under: {DATA_DIR}")
        return

    processed = 0
    skipped = 0

    # ---- 合算（時間重み付き平均用） ----
    total_duration_sec = 0.0

    sum_ipu_c0 = sum_ipu_dur0 = 0.0
    sum_ipu_c1 = sum_ipu_dur1 = 0.0
    sum_pau_c0 = sum_pau_dur0 = 0.0
    sum_pau_c1 = sum_pau_dur1 = 0.0
    sum_gap_c = sum_gap_dur = 0.0
    sum_ov_c = sum_ov_dur = 0.0

    # ---- 単純平均用（各ファイルの“1分あたり”率の平均） ----
    pm_ipu_c0 = []
    pm_ipu_d0 = []
    pm_ipu_c1 = []
    pm_ipu_d1 = []
    pm_pau_c0 = []
    pm_pau_d0 = []
    pm_pau_c1 = []
    pm_pau_d1 = []
    pm_gap_c = []
    pm_gap_d = []
    pm_ov_c = []
    pm_ov_d = []

    for path in files:
        try:
            audio, sr = sf.read(path)
        except Exception as e:
            skipped += 1
            continue

        # ステレオ前提
        if audio.ndim != 2 or audio.shape[1] != 2:
            skipped += 1
            continue

        duration_sec = len(audio) / sr
        if duration_sec <= 0:
            skipped += 1
            continue

        ch0 = audio[:, 0]
        ch1 = audio[:, 1]

        # IPU（+結合）
        ipus_0, ipu_c0, ipu_dur0 = extract_ipu_and_pause(ch0, sr)
        ipus_1, ipu_c1, ipu_dur1 = extract_ipu_and_pause(ch1, sr)

        # Mutual Pause
        pau_c0, pau_dur0, _ = compute_mutual_pause(ipus_0, ipus_1)
        pau_c1, pau_dur1, _ = compute_mutual_pause(ipus_1, ipus_0)

        # Gap / Overlap
        gap_c, gap_dur, ov_c, ov_dur = compute_gap_overlap(ipus_0, ipus_1)

        # ---- 合算（時間重み付き平均用） ----
        total_duration_sec += duration_sec

        sum_ipu_c0 += ipu_c0
        sum_ipu_dur0 += ipu_dur0
        sum_ipu_c1 += ipu_c1
        sum_ipu_dur1 += ipu_dur1

        sum_pau_c0 += pau_c0
        sum_pau_dur0 += pau_dur0
        sum_pau_c1 += pau_c1
        sum_pau_dur1 += pau_dur1

        sum_gap_c += gap_c
        sum_gap_dur += gap_dur
        sum_ov_c += ov_c
        sum_ov_dur += ov_dur

        # ---- 単純平均用（各ファイルの1分あたり） ----
        pm_ipu_c0.append(per_minute(ipu_c0, duration_sec))
        pm_ipu_d0.append(per_minute(ipu_dur0, duration_sec))
        pm_ipu_c1.append(per_minute(ipu_c1, duration_sec))
        pm_ipu_d1.append(per_minute(ipu_dur1, duration_sec))

        pm_pau_c0.append(per_minute(pau_c0, duration_sec))
        pm_pau_d0.append(per_minute(pau_dur0, duration_sec))
        pm_pau_c1.append(per_minute(pau_c1, duration_sec))
        pm_pau_d1.append(per_minute(pau_dur1, duration_sec))

        pm_gap_c.append(per_minute(gap_c, duration_sec))
        pm_gap_d.append(per_minute(gap_dur, duration_sec))
        pm_ov_c.append(per_minute(ov_c, duration_sec))
        pm_ov_d.append(per_minute(ov_dur, duration_sec))

        processed += 1

    # ======= 出力 =======
    print(f"Scanned wavs: {len(files)}, processed(stereo): {processed}, skipped: {skipped}")
    if processed == 0 or total_duration_sec == 0:
        return

    total_minutes = total_duration_sec / 60.0

    # ---- 時間重み付き平均（=合算の1分あたり率） ----
    w_ipu_c0 = sum_ipu_c0 / total_minutes
    w_ipu_d0 = sum_ipu_dur0 / total_minutes
    w_ipu_c1 = sum_ipu_c1 / total_minutes
    w_ipu_d1 = sum_ipu_dur1 / total_minutes

    w_pau_c0 = sum_pau_c0 / total_minutes
    w_pau_d0 = sum_pau_dur0 / total_minutes
    w_pau_c1 = sum_pau_c1 / total_minutes
    w_pau_d1 = sum_pau_dur1 / total_minutes

    w_gap_c = sum_gap_c / total_minutes
    w_gap_d = sum_gap_dur / total_minutes
    w_ov_c = sum_ov_c / total_minutes
    w_ov_d = sum_ov_dur / total_minutes

    # ---- 単純平均（ファイル平均） ----
    def avg(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    a_ipu_c0 = avg(pm_ipu_c0); a_ipu_d0 = avg(pm_ipu_d0)
    a_ipu_c1 = avg(pm_ipu_c1); a_ipu_d1 = avg(pm_ipu_d1)

    a_pau_c0 = avg(pm_pau_c0); a_pau_d0 = avg(pm_pau_d0)
    a_pau_c1 = avg(pm_pau_c1); a_pau_d1 = avg(pm_pau_d1)

    a_gap_c = avg(pm_gap_c);  a_gap_d  = avg(pm_gap_d)
    a_ov_c  = avg(pm_ov_c);   a_ov_d   = avg(pm_ov_d)

    print("\n===== Weighted averages (over all audio minutes) =====")
    print("🗣️ Ch0  IPU   : count/min = {:.3f}, dur/min = {:.3f} sec".format(w_ipu_c0, w_ipu_d0))
    print("🗣️ Ch0  Pause : count/min = {:.3f}, dur/min = {:.3f} sec".format(w_pau_c0, w_pau_d0))
    print("🗣️ Ch1  IPU   : count/min = {:.3f}, dur/min = {:.3f} sec".format(w_ipu_c1, w_ipu_d1))
    print("🗣️ Ch1  Pause : count/min = {:.3f}, dur/min = {:.3f} sec".format(w_pau_c1, w_pau_d1))
    print("🎯 Gap        : count/min = {:.3f}, dur/min = {:.3f} sec".format(w_gap_c, w_gap_d))
    print("🎯 Overlap    : count/min = {:.3f}, dur/min = {:.3f} sec".format(w_ov_c, w_ov_d))

    print("\n===== Simple averages (per-file 1/min averages) =====")
    print("🗣️ Ch0  IPU   : count/min = {:.3f}, dur/min = {:.3f} sec".format(a_ipu_c0, a_ipu_d0))
    print("🗣️ Ch0  Pause : count/min = {:.3f}, dur/min = {:.3f} sec".format(a_pau_c0, a_pau_d0))
    print("🗣️ Ch1  IPU   : count/min = {:.3f}, dur/min = {:.3f} sec".format(a_ipu_c1, a_ipu_d1))
    print("🗣️ Ch1  Pause : count/min = {:.3f}, dur/min = {:.3f} sec".format(a_pau_c1, a_pau_d1))
    print("🎯 Gap        : count/min = {:.3f}, dur/min = {:.3f} sec".format(a_gap_c, a_gap_d))
    print("🎯 Overlap    : count/min = {:.3f}, dur/min = {:.3f} sec".format(a_ov_c, a_ov_d))


if __name__ == "__main__":
    main()
