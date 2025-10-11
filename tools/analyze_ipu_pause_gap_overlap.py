import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

# ===== パラメータ =====
AUDIO_DIR = "data_audio/tmp_csj_20s_head50_1219655.pbs1"
THRESH_DB = -40  # 発話検出の閾値 [dB]
MIN_SILENCE = 0.2  # 無音とみなす最小時間 [s]
SAMPLE_RATE = 16000


def detect_ipus(y, sr, thresh_db=-40, min_silence=0.2):
    """
    音声波形からIPU区間を抽出する関数
    """
    energy = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max
    )
    frame_energy = np.mean(energy, axis=0)
    times = librosa.frames_to_time(np.arange(len(frame_energy)), sr=sr, hop_length=160)

    voiced = frame_energy > thresh_db
    segments = []
    start, active = None, False
    for i, v in enumerate(voiced):
        t = times[i]
        if v and not active:
            start = t
            active = True
        elif not v and active:
            end = t
            if end - start > 0.05:
                segments.append((start, end))
            active = False

    if active:
        segments.append((start, times[-1]))

    # 0.2秒以上の無音で区切る
    merged = []
    prev_end = None
    for seg in segments:
        if prev_end is None:
            merged.append(list(seg))
        elif seg[0] - prev_end >= min_silence:
            merged.append(list(seg))
        else:
            merged[-1][1] = seg[1]
        prev_end = seg[1]
    return merged


def compute_pause_gap_overlap(ipu_a, ipu_b):
    """
    2話者間のPause, Gap, Overlapを算出
    """
    pauses, gaps, overlaps = [], [], []
    for i in range(1, len(ipu_a)):
        pauses.append(ipu_a[i][0] - ipu_a[i - 1][1])

    for a in ipu_a:
        for b in ipu_b:
            # Overlap
            overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
            if overlap > 0:
                overlaps.append(overlap)
            # Gap
            if a[1] < b[0]:
                gaps.append(b[0] - a[1])
            elif b[1] < a[0]:
                gaps.append(a[0] - b[1])

    return pauses, gaps, overlaps


# ===== メイン処理 =====
results = []
for wav in tqdm(sorted(os.listdir(AUDIO_DIR))):
    if not wav.endswith(".wav"):
        continue
    path = os.path.join(AUDIO_DIR, wav)
    y, sr = sf.read(path, always_2d=True)
    yA, yB = y[:, 0], y[:, 1]  # 話者Aと話者B

    ipu_A = detect_ipus(yA, sr, thresh_db=THRESH_DB, min_silence=MIN_SILENCE)
    ipu_B = detect_ipus(yB, sr, thresh_db=THRESH_DB, min_silence=MIN_SILENCE)

    pauses_A, gaps, overlaps = compute_pause_gap_overlap(ipu_A, ipu_B)

    results.append(
        {
            "filename": wav,
            "n_ipu_A": len(ipu_A),
            "n_ipu_B": len(ipu_B),
            "mean_pause_A": np.mean(pauses_A) if pauses_A else 0,
            "mean_gap": np.mean(gaps) if gaps else 0,
            "mean_overlap": np.mean(overlaps) if overlaps else 0,
        }
    )

df = pd.DataFrame(results)
df.to_csv("ipu_pause_gap_overlap_summary.csv", index=False)
print(df.head())
