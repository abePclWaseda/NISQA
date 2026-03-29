import pandas as pd
import sys
import os

# コマンドライン引数からCSVファイルパスを取得（デフォルト値も設定）
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    # デフォルトパス
    csv_path = "/home/acg17145sv/experiments/0162_dialogue_model/NISQA/data_callhome_test/moshi-ccast-bs16-tlr2e-6-dlr4e-6-textpad0.5/NISQA_results.csv"

# CSVファイルの存在確認
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found: {csv_path}")
    sys.exit(1)

# CSV読み込み
df = pd.read_csv(csv_path)

# NaNを除いた有効スコア数と平均値を計算
valid_mos = df["mos_pred"].dropna()
count_mos = len(valid_mos)
mean_mos = valid_mos.mean()

print(f"Average MOS: {mean_mos:.2f}")
print(f"Number of MOS values: {count_mos}")
