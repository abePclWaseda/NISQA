import pandas as pd

# CSV読み込み
df = pd.read_csv("data_sample_audio_nisqa/tabidachi/NISQA_results.csv")

# NaNを除いた有効スコア数と平均値を計算
valid_mos = df["mos_pred"].dropna()
count_mos = len(valid_mos)
mean_mos = valid_mos.mean()

print(f"Average MOS: {mean_mos:.2f}")
print(f"Number of MOS values: {count_mos}")
