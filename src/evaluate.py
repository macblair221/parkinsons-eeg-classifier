import pandas as pd
import numpy as np
from model import train_evaluation_pipeline, loso_evaluation, shuffled_label_control

df = pd.read_csv('features.csv')
print(f"Loaded {df.shape[0]} epochs from {df['subject_id'].nunique()} subjects\n")

train_evaluation_pipeline(df)
observed = float(np.mean(loso_evaluation(df)))
shuffled_label_control(df, n_repeats=100, observed=observed)

clean = df[~df['subject_id'].isin(['pd6', 'pd16'])]
print(f"\n--- Excluding EEGLAB-flagged subjects ({clean['subject_id'].nunique()} remain) ---")
loso_evaluation(clean)

gamma_cols = [c for c in df.columns if c.startswith('gamma_power_')]
print("\n--- Mean gamma share by channel (ON vs OFF) ---")
for c in sorted(gamma_cols):
    off = df[df.medication_state == 0][c].mean()
    on = df[df.medication_state == 1][c].mean()
    print(f"  {c.replace('gamma_power_',''):>4}  OFF {off:.4f}  ON {on:.4f}  diff {off-on:+.4f}")