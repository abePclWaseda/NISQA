import pandas as pd

df = pd.read_csv("data_real/csj_20s_head50/NISQA_results.csv")
mean_mos = df["mos_pred"].mean()
print("Average MOS:", mean_mos)
