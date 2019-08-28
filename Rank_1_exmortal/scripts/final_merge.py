import os

import pandas as pd

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

random_seeds = [21, 22, 23, 24, 25, 26]

final_df = None

for random_seed in random_seeds:
    df = pd.read_csv(os.path.join(home, "results", f"xlnet-base-cased_{random_seed}_1380_seed.csv"))
    if final_df is None:
        final_df = df
    else:
        final_df = pd.merge(final_df, df, on="unique_hash")

for i in range(3):
    final_df[f"class_{i}"] = final_df[[f"{random_seed}_{i}" for random_seed in random_seeds]].mean(axis=1)

final_df = final_df[["unique_hash", "class_0", "class_1", "class_2"]]

final_df["sentiment"] = final_df[[f"class_{i}" for i in range(3)]].idxmax(axis=1)
final_df["sentiment"] = final_df["sentiment"].apply(lambda x: int(x.split("_")[1]))
final_df = final_df[["unique_hash", "sentiment"]]

final_df.to_csv(os.path.join(home, "submissions", "sub.csv"), index=False)

