import pandas as pd

df = pd.read_csv("data/ground_truth/NISQA_results.csv")
mean_mos = df["mos_pred"].mean()
print("Average MOS:", mean_mos)
