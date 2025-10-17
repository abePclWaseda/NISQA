#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ステレオ（各チャネル=各話者）前提のディレクトリ一括計測:
- IPU: 同一話者の非無音区間（同一話者内ギャップが min_silence 未満なら結合）
- Pause: 同一話者の連続IPU間に存在する「純粋な無音（全チャネル無音）」時間
- Gap: 異話者IPU間に存在する「純粋な無音」時間
- Overlap: 複数話者同時発話の時間
各WAVについて per-file CSVを出力し、0–20s 窓で平均表を作成。
"""

import os
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa


# ========= 基本データ構造 =========

@dataclass(frozen=True)
class Seg:
    start: float
    end: float
    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ========= ユーティリティ =========

def merge_short_gaps(segments: List[Seg], min_silence: float) -> List[Seg]:
    """同一話者内でギャップが min_silence 未満なら連結して1つのIPUにする"""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: (s.start, s.end))
    merged = [segs[0]]
    for seg in segs[1:]:
        prev = merged[-1]
        gap = seg.start - prev.end
        if gap < min_silence:  # しきい値未満は結合
            merged[-1] = Seg(prev.start, max(prev.end, seg.end))
        else:
            merged.append(seg)
    return merged


def split_non_silent(y: np.ndarray, sr: int, top_db: float) -> List[Seg]:
    """librosa.effects.split で非無音区間を抽出"""
    intervals = librosa.effects.split(y, top_db=top_db)
    return [Seg(s / sr, e / sr) for s, e in intervals]


def build_atomic(ipus_by_spk: Dict[str, List[Seg]]) -> List[Tuple[float, float, Set[str]]]:
    """全話者のIPU境界を集約し、[s,e) × active_speakers を列挙"""
    boundaries = set()
    for ipus in ipus_by_spk.values():
        for seg in ipus:
            boundaries.add(seg.start)
            boundaries.add(seg.end)
    b = sorted(boundaries)
    atomic = []
    for i in range(len(b) - 1):
        s, e = b[i], b[i + 1]
        if e <= s:
            continue
        active = set()
        for spk, ipus in ipus_by_spk.items():
            for seg in ipus:
                if not (e <= seg.start or seg.end <= s):
                    inter_s, inter_e = max(s, seg.start), min(e, seg.end)
                    if inter_e - inter_s > 0:
                        active.add(spk); break
        atomic.append((s, e, active))
    return atomic


def union_silence(atomic: List[Tuple[float, float, Set[str]]]) -> List[Seg]:
    """原子的区間のうち active=set() を連結した無音タイムライン"""
    silent = []
    cur_s, cur_e = None, None
    for s, e, active in atomic:
        if len(active) == 0:
            if cur_s is None:
                cur_s, cur_e = s, e
            else:
                if math.isclose(s, cur_e) or s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    silent.append(Seg(cur_s, cur_e))
                    cur_s, cur_e = s, e
        else:
            if cur_s is not None:
                silent.append(Seg(cur_s, cur_e))
                cur_s, cur_e = None, None
    if cur_s is not None:
        silent.append(Seg(cur_s, cur_e))
    return silent


def intersect(a: Seg, b: Seg) -> float:
    """2区間の交差長"""
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def intersect_list_with(seg_list: List[Seg], cut: Seg) -> List[Seg]:
    out = []
    for s in seg_list:
        d = intersect(s, cut)
        if d > 0:
            out.append(Seg(max(s.start, cut.start), min(s.end, cut.end)))
    return out


def total_duration(segs: List[Seg]) -> float:
    return sum(s.duration for s in segs)


def clip_interval_pairs(pairs: List[Tuple[float, float]], w0: float, w1: float) -> List[Seg]:
    out = []
    for s, e in pairs:
        a, b = max(s, w0), min(e, w1)
        if b > a:
            out.append(Seg(a, b))
    return out


def union_duration(pairs: List[Tuple[float, float]], w0: float, w1: float) -> float:
    """[w0,w1) 内で複数区間のユニオン長"""
    segs = clip_interval_pairs(pairs, w0, w1)
    if not segs:
        return 0.0
    segs.sort(key=lambda s: (s.start, s.end))
    cur = segs[0]
    acc = 0.0
    for s in segs[1:]:
        if s.start <= cur.end or math.isclose(s.start, cur.end):
            cur = Seg(cur.start, max(cur.end, s.end))
        else:
            acc += cur.duration
            cur = s
    acc += cur.duration
    return acc


# ========= 指標計算 =========

def compute_ipus_per_channel(y2d: np.ndarray, sr: int, top_db: float, min_silence: float) -> Dict[str, List[Seg]]:
    """
    各チャネルを1話者として、非無音→短ギャップ結合でIPUを作成
    """
    n_ch = y2d.shape[1]
    out: Dict[str, List[Seg]] = {}
    for ch in range(n_ch):
        spk = f"spk_{ch}"
        ipu_raw = split_non_silent(y2d[:, ch].astype(np.float32), sr, top_db)
        ipu_merged = merge_short_gaps(ipu_raw, min_silence)
        out[spk] = ipu_merged
    return out


def compute_pauses(ipus_by_spk: Dict[str, List[Seg]], silent_union: List[Seg]) -> List[Dict]:
    rows = []
    for spk, ipus in ipus_by_spk.items():
        ipus = sorted(ipus, key=lambda s: s.start)
        for a, b in zip(ipus, ipus[1:]):
            mid = Seg(a.end, b.start)
            if mid.duration <= 0:
                continue
            silence_only = total_duration(intersect_list_with(silent_union, mid))
            if silence_only > 0:
                rows.append({
                    "speaker": spk,
                    "prev_ipu_start": a.start,
                    "prev_ipu_end": a.end,
                    "next_ipu_start": b.start,
                    "next_ipu_end": b.end,
                    "pause_silence_sec": silence_only,
                })
    return rows


def compute_gaps(ipus_by_spk: Dict[str, List[Seg]], silent_union: List[Seg]) -> List[Dict]:
    all_ipus = []
    for spk, ipus in ipus_by_spk.items():
        for seg in ipus:
            all_ipus.append((seg, spk))
    all_ipus.sort(key=lambda x: (x[0].start, x[0].end))
    rows = []
    for (seg_a, spk_a), (seg_b, spk_b) in zip(all_ipus, all_ipus[1:]):
        if spk_a == spk_b:
            continue
        mid = Seg(seg_a.end, seg_b.start)
        if mid.duration <= 0:
            continue
        silence_only = total_duration(intersect_list_with(silent_union, mid))
        if silence_only > 0:
            rows.append({
                "from_speaker": spk_a,
                "to_speaker": spk_b,
                "prev_ipu_start": seg_a.start,
                "prev_ipu_end": seg_a.end,
                "next_ipu_start": seg_b.start,
                "next_ipu_end": seg_b.end,
                "gap_silence_sec": silence_only,
            })
    return rows


def compute_overlaps(atomic: List[Tuple[float, float, Set[str]]]) -> List[Dict]:
    rows = []
    cur_s, cur_e = None, None
    for s, e, active in atomic:
        if len(active) >= 2:
            if cur_s is None:
                cur_s, cur_e = s, e
            else:
                if math.isclose(s, cur_e) or s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    rows.append({"start": cur_s, "end": cur_e, "duration": max(0.0, cur_e - cur_s)})
                    cur_s, cur_e = s, e
        else:
            if cur_s is not None:
                rows.append({"start": cur_s, "end": cur_e, "duration": max(0.0, cur_e - cur_s)})
                cur_s, cur_e = None, None
    if cur_s is not None:
        rows.append({"start": cur_s, "end": cur_e, "duration": max(0.0, cur_e - cur_s)})
    return rows


# ========= 出力 =========

def df_safe_write(df: pd.DataFrame, path: Path, columns: List[str], sort_by: List[str] = None):
    if df is None or df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        missing = [c for c in columns if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        if sort_by:
            exist = [c for c in sort_by if c in df.columns]
            if exist:
                df = df.sort_values(exist)
    df.to_csv(path, index=False)


def write_perfile_outputs(outdir: Path,
                          ipus_by_spk: Dict[str, List[Seg]],
                          pauses: List[Dict],
                          gaps: List[Dict],
                          overlaps: List[Dict],
                          audio_path: str,
                          sr: int,
                          nsec: float):
    outdir.mkdir(parents=True, exist_ok=True)

    # IPUs
    rows = []
    for spk, ipus in ipus_by_spk.items():
        for seg in ipus:
            rows.append({"speaker": spk, "start": seg.start, "end": seg.end, "duration": seg.duration})
    df_ipu = pd.DataFrame(rows)
    df_safe_write(df_ipu, outdir / "ipus.csv", ["speaker", "start", "end", "duration"], ["start", "end"])

    # Pauses
    df_p = pd.DataFrame(pauses)
    df_safe_write(
        df_p,
        outdir / "pauses.csv",
        ["speaker", "prev_ipu_start", "prev_ipu_end", "next_ipu_start", "next_ipu_end", "pause_silence_sec"],
        ["prev_ipu_end"],
    )

    # Gaps
    df_g = pd.DataFrame(gaps)
    df_safe_write(
        df_g,
        outdir / "gaps.csv",
        ["from_speaker", "to_speaker", "prev_ipu_start", "prev_ipu_end", "next_ipu_start", "next_ipu_end", "gap_silence_sec"],
        ["prev_ipu_end"],
    )

    # Overlaps
    df_o = pd.DataFrame(overlaps)
    df_safe_write(df_o, outdir / "overlaps.csv", ["start", "end", "duration"], ["start"])

    # Summary
    total_overlap = float(df_o["duration"].sum()) if not df_o.empty else 0.0
    total_pause = float(df_p["pause_silence_sec"].sum()) if not df_p.empty else 0.0
    total_gap = float(df_g["gap_silence_sec"].sum()) if not df_g.empty else 0.0
    n_ipu = sum(len(v) for v in ipus_by_spk.values())

    lines = []
    lines.append(f"Audio: {audio_path}")
    lines.append(f"SampleRate: {sr}")
    lines.append(f"Duration (sec): {nsec:.3f}")
    lines.append(f"Speakers (channels): {len(ipus_by_spk)}  -> {', '.join(sorted(ipus_by_spk.keys()))}")
    lines.append(f"Total IPUs: {n_ipu}")
    for spk, ipus in ipus_by_spk.items():
        dur = sum(s.duration for s in ipus)
        lines.append(f"  - {spk}: IPUs={len(ipus)}, total_speaking_sec={dur:.3f}")
    lines.append(f"Total Pause (same-speaker, pure silence): {total_pause:.3f} sec")
    lines.append(f"Total Gap (cross-speaker, pure silence): {total_gap:.3f} sec")
    lines.append(f"Total Overlap (multi-speaker): {total_overlap:.3f} sec")

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ========= 0–20s 集計（IPUは union / sum_spk 切替可）=========

def collect_windowed_0_20(outdir: Path, ipu_mode: str = "union") -> Dict[str, float]:
    ipus_csv = outdir / "ipus.csv"
    pauses_csv = outdir / "pauses.csv"
    gaps_csv = outdir / "gaps.csv"
    overlaps_csv = outdir / "overlaps.csv"

    W0, W1 = 0.0, 20.0

    # IPU
    ipu_sec = 0.0
    if ipus_csv.exists():
        df = pd.read_csv(ipus_csv)
        if not df.empty:
            if ipu_mode == "sum_spk":
                s = 0.0
                for _, r in df.iterrows():
                    s += intersect(Seg(float(r["start"]), float(r["end"])), Seg(W0, W1))
                ipu_sec = s
            else:
                pairs = [(float(r["start"]), float(r["end"])) for _, r in df.iterrows()]
                ipu_sec = union_duration(pairs, W0, W1)

    # Pause
    pause_sec = 0.0
    if pauses_csv.exists():
        dfp = pd.read_csv(pauses_csv)
        if not dfp.empty:
            for _, r in dfp.iterrows():
                pause_sec += intersect(Seg(float(r["prev_ipu_end"]), float(r["next_ipu_start"])), Seg(W0, W1))

    # Gap
    gap_sec = 0.0
    if gaps_csv.exists():
        dfg = pd.read_csv(gaps_csv)
        if not dfg.empty:
            for _, r in dfg.iterrows():
                gap_sec += intersect(Seg(float(r["prev_ipu_end"]), float(r["next_ipu_start"])), Seg(W0, W1))

    # Overlap
    overlap_sec = 0.0
    if overlaps_csv.exists():
        dfo = pd.read_csv(overlaps_csv)
        if not dfo.empty:
            for _, r in dfo.iterrows():
                overlap_sec += intersect(Seg(float(r["start"]), float(r["end"])), Seg(W0, W1))

    return {
        "ipu_0_20": ipu_sec,
        "pause_0_20": pause_sec,
        "gap_0_20": gap_sec,
        "overlap_0_20": overlap_sec,
    }


# ========= メイン処理 =========

def process_one_wav(wav: Path, out_root: Path, min_silence: float, top_db: float):
    y, sr = sf.read(wav, always_2d=True)  # (n_samples, n_channels)
    nsec = len(y) / sr

    ipus_by_spk = compute_ipus_per_channel(y, sr, top_db=top_db, min_silence=min_silence)
    atomic = build_atomic(ipus_by_spk)
    silent = union_silence(atomic)

    pauses = compute_pauses(ipus_by_spk, silent)
    gaps = compute_gaps(ipus_by_spk, silent)
    overlaps = compute_overlaps(atomic)

    outdir = out_root / wav.stem
    write_perfile_outputs(outdir, ipus_by_spk, pauses, gaps, overlaps, str(wav), sr, nsec)


def main():
    parser = argparse.ArgumentParser(description="Batch IPU/Pause/Gap/Overlap for stereo (each channel = speaker).")
    parser.add_argument("--audio_dir", type=str, required=True, help="WAV directory (20s stereo files)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output base directory")
    parser.add_argument("--min_silence", type=float, default=0.2, help="IPU分割の最小無音秒（短ければ結合）")
    parser.add_argument("--top_db", type=float, default=40.0, help="無音判定（librosa.effects.split の dB）")
    parser.add_argument("--ipu_mode", type=str, default="union", choices=["union", "sum_spk"], help="IPUの合算方法")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files in {audio_dir}")

    # 1) 各WAVを処理
    for wav in wavs:
        print(f"[Process] {wav}")
        try:
            process_one_wav(wav, out_root, args.min_silence, args.top_db)
        except Exception as e:
            print(f"[WARN] failed: {wav} ({e})")

    # 2) 0–20s 集計表
    rows = []
    for wav in wavs:
        subdir = out_root / wav.stem
        if not subdir.exists():
            continue
        m = collect_windowed_0_20(subdir, ipu_mode=args.ipu_mode)
        rows.append({
            "file": wav.stem,
            "IPU(0-20s)": m["ipu_0_20"],
            "Pause(0-20s)": m["pause_0_20"],
            "Gap(0-20s)": m["gap_0_20"],
            "Overlap(0-20s)": m["overlap_0_20"],
        })

    df = pd.DataFrame(rows).sort_values("file")
    # 比率列（分母は20秒）
    for k in ["IPU(0-20s)", "Pause(0-20s)", "Gap(0-20s)", "Overlap(0-20s)"]:
        df[k] = df[k].fillna(0.0)
        df[f"{k}_ratio"] = (df[k] / 20.0).round(6)

    # 平均行
    mean_row = df.mean(numeric_only=True).to_dict()
    df.loc[len(df)] = {"file": "MEAN(0-20s)", **mean_row}

    out_csv = out_root / "table_0_20.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Done] per-file outputs under: {out_root}")
    print(f"[Done] table saved: {out_csv}")
    if not df.empty:
        print(df.tail(1))


if __name__ == "__main__":
    main()
