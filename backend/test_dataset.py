import pandas as pd
df = pd.read_csv("data/raw/filtered_nowebatt.csv", low_memory=False)
df.columns = df.columns.str.strip()
label_col = None
for c in df.columns:
    if c.strip().upper() in ("LABEL", "CLASS", "ATTACK"):
        label_col = c
        break

print("Label column:", label_col)
if label_col:
    print(df[label_col].value_counts())
