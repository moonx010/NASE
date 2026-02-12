"""Summarize all evaluation results from enhanced/ directory as markdown table.

Usage:
    python scripts/summarize_results.py                          # scan all enhanced/
    python scripts/summarize_results.py enhanced                 # same
    python scripts/summarize_results.py enhanced/e3-adaptive-*   # glob specific dirs
"""
import os
import sys
import pandas as pd
from glob import glob
from datetime import datetime

# Collect target dirs from args (supports shell glob)
args = sys.argv[1:] if len(sys.argv) > 1 else ["enhanced"]

csv_files = []
for arg in args:
    csv_files.extend(sorted(glob(os.path.join(arg, "**", "_results.csv"), recursive=True)))

if not csv_files:
    print("No _results.csv files found")
    exit()

# Build filename from model names found
model_names = set()
groups = {}
for csv_path in csv_files:
    # Parse: enhanced/<model>/<test_set>/_results.csv
    parts = csv_path.split("/")
    idx = parts.index("enhanced") if "enhanced" in parts else 0
    model = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
    test_set = parts[idx + 2] if idx + 2 < len(parts) else "unknown"
    # Strip _results.csv from test_set if it ended up there
    test_set = test_set.replace("_results.csv", "").strip("/")
    if not test_set or test_set == model:
        test_set = "unknown"

    model_names.add(model)
    df = pd.read_csv(csv_path)
    groups.setdefault(test_set, []).append({
        "Model": model,
        "PESQ": df['pesq'].mean(),
        "ESTOI": df['estoi'].mean(),
        "SI-SDR": df['si_sdr'].mean(),
        "SI-SIR": df['si_sir'].mean(),
        "SI-SAR": df['si_sar'].mean(),
        "Files": len(df),
    })

# Generate descriptive filename
date_str = datetime.now().strftime("%Y%m%d")
# Shorten model names: find common prefix to use as label
sorted_models = sorted(model_names)
if len(sorted_models) == 1:
    label = sorted_models[0]
else:
    # Find common prefix among model names
    prefix = os.path.commonprefix(sorted_models).rstrip("-_")
    if prefix and len(prefix) >= 2:
        label = prefix
    else:
        label = "_".join(sorted_models[:3])
        if len(sorted_models) > 3:
            label += f"_+{len(sorted_models)-3}more"

output_md = os.path.join("enhanced", f"results_{label}_{date_str}.md")

# Build markdown
lines = [f"# Results: {label} ({date_str})\n"]

for test_set, rows in groups.items():
    n_files = rows[0]["Files"]
    lines.append(f"## {test_set} ({n_files} files)\n")
    lines.append("| Model | PESQ | ESTOI | SI-SDR | SI-SIR | SI-SAR |")
    lines.append("|-------|------|-------|--------|--------|--------|")
    for r in sorted(rows, key=lambda x: x["Model"]):
        lines.append(f"| {r['Model']} | {r['PESQ']:.2f} | {r['ESTOI']:.2f} | {r['SI-SDR']:.1f} | {r['SI-SIR']:.1f} | {r['SI-SAR']:.1f} |")
    lines.append("")

md_text = "\n".join(lines)
print(md_text)

with open(output_md, "w") as f:
    f.write(md_text)
print(f"\nSaved to {output_md}")
