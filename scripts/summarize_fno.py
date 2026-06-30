"""
Temporary script: compute average MSE and relative L2 error for FNO
across three ODE operators in the benchmarks3_reproduction output folder.
Missing results are skipped; averages are computed over available runs only.

Usage:
    python scripts/summarize_fno.py [--dir benchmarks3_reproduction]
"""

import json
import os
import argparse
from collections import defaultdict

OPERATORS = ["Antideriv", "Homogeneous", "Nonlinear"]


def find_fno_metrics(base_dir):
    """Return list of (operator, seed, mse, rel_l2) for all found FNO results."""
    records = []
    for op in OPERATORS:
        op_dir = os.path.join(base_dir, op)
        if not os.path.isdir(op_dir):
            continue
        for exp_name in os.listdir(op_dir):
            if "FNO" not in exp_name:
                continue
            metric_path = os.path.join(op_dir, exp_name, "metric.json")
            if not os.path.isfile(metric_path):
                continue
            with open(metric_path) as f:
                data = json.load(f)
            m = data.get("metrics", {})
            mse = m.get("MSE")
            rel_l2 = m.get("rel_l2")
            if mse is None or rel_l2 is None:
                print(f"  [WARN] Missing MSE or rel_l2 in {metric_path}")
                continue
            records.append((op, exp_name, mse, rel_l2))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="benchmarks3_reproduction",
                        help="Output directory to scan")
    args = parser.parse_args()

    base_dir = args.dir
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    records = find_fno_metrics(base_dir)
    if not records:
        print("No FNO metric.json files found.")
        return

    # Group by operator
    by_op = defaultdict(list)
    for op, name, mse, rel_l2 in records:
        by_op[op].append((name, mse, rel_l2))

    print(f"\n{'='*60}")
    print(f"  FNO Results Summary  ({base_dir})")
    print(f"{'='*60}")

    all_mse, all_rel = [], []

    for op in OPERATORS:
        runs = by_op.get(op, [])
        if not runs:
            print(f"\n  {op:15s}  [NO DATA]")
            continue
        mses   = [r[1] for r in runs]
        rels   = [r[2] for r in runs]
        avg_mse = sum(mses) / len(mses)
        avg_rel = sum(rels) / len(rels)
        all_mse.extend(mses)
        all_rel.extend(rels)

        print(f"\n  {op}")
        print(f"    runs      : {len(runs)}")
        for name, mse, rel in sorted(runs, key=lambda x: x[0]):
            print(f"    {name[-30:]:30s}  MSE={mse:.6f}  rel_L2={rel:.4%}")
        print(f"    ── avg MSE : {avg_mse:.6f}")
        print(f"    ── avg rel : {avg_rel:.4%}")

    if all_mse:
        print(f"\n{'─'*60}")
        print(f"  Overall ({len(all_mse)} runs across {len(by_op)} operator(s))")
        print(f"    avg MSE : {sum(all_mse)/len(all_mse):.6f}")
        print(f"    avg rel : {sum(all_rel)/len(all_rel):.4%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
