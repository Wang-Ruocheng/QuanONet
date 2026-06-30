"""
Summarize FNO results from a reproduction output folder.

Outputs:
  - Console summary (avg MSE / rel_l2 per operator)
  - Excel file with one row per (operator, model, seed, metric):
      columns: operator | model | seed | metric | value
      rel_err is multiplied by 100 (percentage points)

Usage:
    python scripts/summarize_fno.py [--dir benchmarks3_reproduction]
"""

import json
import os
import re
import argparse
from collections import defaultdict

OPERATORS = ["Antideriv", "Homogeneous", "Nonlinear"]


def parse_seed(exp_name):
    m = re.search(r"[Ss]eed(\d+)", exp_name)
    return int(m.group(1)) if m else -1


def find_fno_metrics(base_dir):
    """Return list of (operator, seed, mse, rel_l2) for all found FNO results."""
    records = []
    for op in OPERATORS:
        op_dir = os.path.join(base_dir, op)
        if not os.path.isdir(op_dir):
            continue
        for exp_name in sorted(os.listdir(op_dir)):
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
            seed = parse_seed(exp_name)
            records.append((op, seed, mse, rel_l2))
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

    # ── Console summary ───────────────────────────────────────────────────────
    by_op = defaultdict(list)
    for op, seed, mse, rel_l2 in records:
        by_op[op].append((seed, mse, rel_l2))

    print(f"\n{'='*60}")
    print(f"  FNO Results Summary  ({base_dir})")
    print(f"{'='*60}")

    all_mse, all_rel = [], []
    for op in OPERATORS:
        runs = by_op.get(op, [])
        if not runs:
            print(f"\n  {op:15s}  [NO DATA]")
            continue
        mses = [r[1] for r in runs]
        rels = [r[2] for r in runs]
        all_mse.extend(mses)
        all_rel.extend(rels)
        print(f"\n  {op}  ({len(runs)} runs)")
        for seed, mse, rel in sorted(runs):
            print(f"    Seed{seed}  MSE={mse:.6f}  rel_err={rel*100:.6f}%")
        print(f"    ── avg MSE : {sum(mses)/len(mses):.6f}")
        print(f"    ── avg rel : {sum(rels)/len(rels)*100:.6f}%")

    if all_mse:
        print(f"\n{'─'*60}")
        print(f"  Overall ({len(all_mse)} runs, {len(by_op)} operator(s))")
        print(f"    avg MSE : {sum(all_mse)/len(all_mse):.6f}")
        print(f"    avg rel : {sum(all_rel)/len(all_rel)*100:.6f}%")

    # ── Per-operator mean ± std of rel_err ───────────────────────────────────
    import statistics
    print(f"\n{'─'*60}")
    print(f"  Relative Error (%) per Operator  [mean ± std, 3 d.p.]")
    print(f"{'─'*60}")
    op_means = []
    for op in OPERATORS:
        runs = by_op.get(op, [])
        if not runs:
            print(f"  {op:15s}  [NO DATA]")
            continue
        rels_pct = [r[2] * 100 for r in runs]
        mean = statistics.mean(rels_pct)
        std  = statistics.pstdev(rels_pct) if len(rels_pct) > 1 else 0.0
        op_means.append(mean)
        print(f"  {op:15s}  {mean:.3f}% ± {std:.3f}%  (n={len(rels_pct)})")
    if op_means:
        overall_mean = statistics.mean(op_means)
        overall_std  = statistics.pstdev(op_means) if len(op_means) > 1 else 0.0
        print(f"  {'Overall (op avg)':15s}  {overall_mean:.3f}% ± {overall_std:.3f}%")
    print(f"{'='*60}\n")

    # ── Excel output ──────────────────────────────────────────────────────────
    try:
        import openpyxl
    except ImportError:
        print("[INFO] openpyxl not installed, falling back to CSV.")
        _write_csv(records, base_dir)
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "FNO Results"
    ws.append(["operator", "model", "seed", "metric", "value"])

    for op in OPERATORS:
        runs = by_op.get(op, [])
        for seed, mse, rel_l2 in sorted(runs):
            ws.append([op, "FNO", seed, "MSE",     mse])
            ws.append([op, "FNO", seed, "rel_err", rel_l2 * 100])

    out_path = os.path.join(base_dir, "fno_results.xlsx")
    wb.save(out_path)
    print(f"Excel saved to: {out_path}")


def _write_csv(records, base_dir):
    import csv
    out_path = os.path.join(base_dir, "fno_results.csv")
    by_op = defaultdict(list)
    for op, seed, mse, rel_l2 in records:
        by_op[op].append((seed, mse, rel_l2))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operator", "model", "seed", "metric", "value"])
        for op in OPERATORS:
            for seed, mse, rel_l2 in sorted(by_op.get(op, [])):
                w.writerow([op, "FNO", seed, "MSE",     mse])
                w.writerow([op, "FNO", seed, "rel_err", rel_l2 * 100])
    print(f"CSV saved to: {out_path}")


if __name__ == "__main__":
    main()
