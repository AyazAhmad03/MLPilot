"""Generates iris_messy.csv — a deliberately broken version of iris.csv
to demo MLPilot's data cleaning loop:
  - ~85% missing in one column  -> column dropped
  - ~70% missing in another     -> column dropped
  - 10 missing target values    -> rows dropped
  - 3 duplicate rows            -> duplicates dropped
"""
import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")
rng = np.random.default_rng(42)

# 1. Blank ~85% of one column -> cleaning should DROP this column
mask = rng.random(len(df)) < 0.85
df.loc[mask, df.columns[3]] = np.nan

# 2. Blank ~70% of another column -> also dropped
mask = rng.random(len(df)) < 0.70
df.loc[mask, df.columns[1]] = np.nan

# 3. Make 10 target values missing -> those ROWS should be dropped
target = df.columns[-1]
rows = rng.choice(len(df), 10, replace=False)
df.loc[rows, target] = np.nan

# 4. Duplicate 3 rows -> cleaning should drop duplicates
df = pd.concat([df, df.iloc[[5, 20, 40]]], ignore_index=True)

df.to_csv("iris_messy.csv", index=False)

missing = df.isnull().sum().sum()
print(f"messy csv saved: iris_messy.csv")
print(f"total missing cells: {missing} / {df.size} ({missing/df.size:.1%})")