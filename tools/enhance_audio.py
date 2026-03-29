#!/usr/bin/env python
"""
DeepFilterNetを使って音声ファイルを高品質化するスクリプト
"""
import os
from pathlib import Path
from df.enhance import enhance, init_df, load_audio, save_audio

# 入出力ディレクトリ
INPUT_DIR = Path("/groups/gcg51557/experiments/0162_dialogue_model/NISQA/data_sample_audio/callhome")
OUTPUT_DIR = Path("/groups/gcg51557/experiments/0162_dialogue_model/NISQA/data_sample_audio/callhome_enhanced")

def main():
    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # DeepFilterNet初期化
    print("DeepFilterNet を初期化中...")
    model, df_state, _ = init_df()

    # 音声ファイル一覧取得
    wav_files = sorted(INPUT_DIR.glob("*.wav"))
    print(f"処理対象: {len(wav_files)} ファイル")

    for i, input_path in enumerate(wav_files):
        output_path = OUTPUT_DIR / input_path.name

        if output_path.exists():
            print(f"[{i+1}/{len(wav_files)}] スキップ (既存): {input_path.name}")
            continue

        print(f"[{i+1}/{len(wav_files)}] 処理中: {input_path.name}")

        try:
            # 音声読み込み
            audio, _ = load_audio(str(input_path), sr=df_state.sr())

            # ノイズ除去
            enhanced = enhance(model, df_state, audio)

            # 保存
            save_audio(str(output_path), enhanced, df_state.sr())

        except Exception as e:
            print(f"  エラー: {e}")

    print("完了!")

if __name__ == "__main__":
    main()
