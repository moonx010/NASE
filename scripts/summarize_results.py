"""Summarize all evaluation results from enhanced/ directory."""
import os
import pandas as pd
from glob import glob

enhanced_root = "enhanced"
csv_files = sorted(glob(os.path.join(enhanced_root, "**", "_results.csv"), recursive=True))

if not csv_files:
    print("No _results.csv files found in enhanced/")
    exit()

rows = []
for csv_path in csv_files:
    parts = csv_path.replace(enhanced_root + "/", "").replace("/_results.csv", "")
    tokens = parts.split("/")
    model = tokens[0]
    test_set = tokens[1] if len(tokens) > 1 else "unknown"
    df = pd.read_csv(csv_path)
    rows.append({
        "Model": model,
        "TestSet": test_set,
        "PESQ": f"{df['pesq'].mean():.2f}",
        "ESTOI": f"{df['estoi'].mean():.2f}",
        "SI-SDR": f"{df['si_sdr'].mean():.1f}",
        "SI-SIR": f"{df['si_sir'].mean():.1f}",
        "SI-SAR": f"{df['si_sar'].mean():.1f}",
        "Files": len(df),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print()
summary.to_csv(os.path.join(enhanced_root, "summary.csv"), index=False)
print(f"Saved to {enhanced_root}/summary.csv")
