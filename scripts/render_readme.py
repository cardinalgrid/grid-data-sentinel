"""Rewrite the results block of README.md from results/summary.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- results:start -->", "<!-- results:end -->"

LABELS = {
    "sentinel": "Composite v0.1 (TEDA levels + TEDA differences + stuck rule)",
    "sentinel_v2": "Composite v0.2 (v0.1 detectors confirmed by the calendar profile)",
    "profile_residual": "Profile residual alone (day type, 2 weeks + last-year analogs, k=4)",
    "modified_zscore": "Modified z-score (30 d, k=3.5)",
    "relative_deviation": "Relative deviation from 5-h centred mean (15%)",
    "teda_diff": "Recursive TEDA (differenced, m=3)",
    "teda_level": "Recursive TEDA (levels, m=4)",
    "teda_level_robust": "Recursive TEDA (levels, m=4, robust update)",
    "autoencoder": "Sparse autoencoder (window 4, code 2, factor 20)",
    "rolling_zscore": "Rolling z-score (168 h, k=3)",
    "hampel": "Hampel filter (24 h, k=3)",
    "iqr": "Global IQR (k=1.5)",
}


def main() -> int:
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    det = summary["detectors"]
    head = (
        f"Benchmark v{summary['version']} run {summary['generated'][:10]}: "
        f"{summary['n_series']} BA-years ({', '.join(summary['bas'])}; {', '.join(map(str, summary['years']))}), "
        f"{summary['injection_rate'] * 100:.1f}% of readings corrupted, tolerance ±{summary['tolerance']} reading. "
        "Means over series."
    )
    lines = [
        head,
        "",
        "| Detector | Precision | Precision (adj.) | Recall | F1 (adj.) | MCC | Spike | Dip | Zero | Stuck | Scale | s/series |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in det.items():
        lines.append(
            f"| {LABELS.get(name, name)} | {m['precision']:.2f} | {m['precision_adj']:.2f} | {m['recall']:.2f} | {m['f1_adj']:.2f} | {m['mcc']:.2f} "
            f"| {m.get('recall_spike', float('nan')):.2f} | {m.get('recall_dip', float('nan')):.2f} "
            f"| {m.get('recall_zero', float('nan')):.2f} | {m.get('recall_stuck', float('nan')):.2f} "
            f"| {m.get('recall_scale', float('nan')):.2f} | {m['seconds']:.1f} |"
        )
    lines.append("")
    lines.append("Columns Spike to Scale are recall by anomaly type. Precision (adj.) and F1 (adj.) do not count as false alarms the flags on readings that the public series already had wrong (frozen runs, non-positive values, unit errors; see `docs/real_cases.md`); the plain precision does. Full table: `results/benchmark_by_series.csv`.")
    block = f"{START}\n" + "\n".join(lines) + f"\n{END}"
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text:
        print("markers not found in README.md", file=sys.stderr)
        return 1
    text = re.sub(re.escape(START) + ".*?" + re.escape(END), block.replace("\\", "\\\\"), text, flags=re.DOTALL)
    readme.write_text(text, encoding="utf-8", newline="\n")
    print("README results block updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
