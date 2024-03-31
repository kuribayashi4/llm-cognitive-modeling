import json
import pandas as pd
from typing import List, Dict

file = "data/NS/all.txt.annotation"
df = pd.read_table(file)
times = df["time"].values
mean_time = times[times > 0].mean()
std_time = times[times > 0].std()
idx_long_time = df["time"] > mean_time + 3 * std_time
idx_short_time = df["time"] < mean_time - 3 * std_time
df.loc[idx_long_time, "time"] = 0
# df.loc[idx_long_time, "logtime"] = "-Infinity"
# df.loc[idx_long_time, "invtime"] = "Infinity"
df.loc[idx_short_time, "time"] = 0
# df.loc[idx_short_time, "logtime"] = "-Infinity"
# df.loc[idx_short_time, "invtime"] = "Infinity"
df = df.drop(["Unnamed: 0"], axis=1)
df.to_csv(file + ".filtered.averaged_rt.csv", quoting=2, escapechar="\\")