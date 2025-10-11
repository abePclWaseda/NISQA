import pandas as pd

df = pd.read_csv(
    "data_tabidachi/moshi_stage3_new_jchat_tabidachi/NISQA_results.csv"
)
mean_mos = df["mos_pred"].mean()
print("Average MOS:", mean_mos)
