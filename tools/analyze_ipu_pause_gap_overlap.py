import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

AUDIO_DIR = "/home/acg17145sv/experiments/0162_dialogue_model/NISQA/data_audio/tmp_csj_20s_head50_1219655.pbs1"
MIN_SILENCE = 0.2  # 秒
SAMPLE_RATE = 16000


def detect_ipus_librosa(y, sr, top_db=40, min_silence=0.2):
    """librosa.effects.split() に基づく IPU 検出"""
    intervals = librosa.effects.split(y, top_db=top_db)
    ipus = []
    for s, e in intervals:
        start, end = s / sr, e / sr
        if len(ipus) > 0 and start - ipus[-1][1] < min_silence:
            ipus[-1][1] = end
        else:
            ipus.append([start, end])
    return ipus


def compute_pause_gap_overlap(ipu_a, ipu_b):
    pauses, gaps, overlaps = [], [], []
    # Pause（同一話者内）
    for i in range(1, len(ipu_a)):
        pauses.append(ipu_a[i][0] - ipu_a[i - 1][1])

    # Gap / Overlap（話者間）
    for a in ipu_a:
        for b in ipu_b:
            overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
            if overlap > 0:
                overlaps.append(overlap)
            if a[1] < b[0]:
                gaps.append(b[0] - a[1])
            elif b[1] < a[0]:
                gaps.append(a[0] - b[1])
    return pauses, gaps, overlaps


results = []
for wav in tqdm(sorted(os.listdir(AUDIO_DIR))):
    if not wav.endswith(".wav"):
        continue
    path = os.path.join(AUDIO_DIR, wav)
    y, sr = sf.read(path, always_2d=True)
    yA, yB = y[:, 0], y[:, 1]  # L=話者A, R=話者B

    ipu_A = detect_ipus_librosa(yA, sr, top_db=40, min_silence=MIN_SILENCE)
    ipu_B = detect_ipus_librosa(yB, sr, top_db=40, min_silence=MIN_SILENCE)

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
df.to_csv("ipu_pause_gap_overlap_refined.csv", index=False)
print(df.head())
