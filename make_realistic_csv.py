import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "customer_id": range(1, n + 1),                     # -> should be DROPPED
    "signup_date": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),  # -> EXPANDED
    "age": rng.integers(18, 70, n),
    "monthly_spend": rng.normal(50, 15, n).round(2),
    "churn": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
})
df.to_csv("customers.csv", index=False)
print("saved customers.csv")