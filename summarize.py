"""Aggregate runs/*/metrics.json into mean +- sd across seeds for each (model, condition, wd) cell.

    python summarize.py --runs runs --out runs/summary.csv
    python summarize.py --runs runs --by-split      # keep split seeds as separate rows (different held-out sets)

Selection note: 'val_selected_test' is the TEST accuracy at the epoch chosen by VALIDATION accuracy.
Never tune on the test columns.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

COLS = [("train", lambda r: r["final"]["train"]["accuracy"]),
        ("id", lambda r: r["final"].get("id", {}).get("accuracy")),
        ("val", lambda r: r["final"]["val"]["accuracy"]),
        ("test", lambda r: r["final"]["test"]["accuracy"]),
        ("test_per_function", lambda r: r["breakdown_test"]["unique_function_mean"]),
        ("test_functions_solved_90", lambda r: r["breakdown_test"]["functions_solved_ge_90pct"]),
        ("test_CE", lambda r: r["final"]["test"]["cross_entropy"]),
        ("test_ECE", lambda r: r["final"]["test"]["ece"]),
        ("val_selected_test", lambda r: (r.get("selected_by_val") or {}).get("test")),
        ("epoch_id_90", lambda r: r["first_epoch_ge_90"].get("id")),
        ("epoch_val_50", lambda r: r["first_epoch_ge_50"].get("val")),
        ("test_rect_matched", lambda r: r["baselines"]["test"]["rectangle_solver_matched_cells"]["accuracy"]),
        ("d_second_div", lambda r: r["final"]["d_regions"].get("d_second_div"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--by-split", action="store_true")
    args = ap.parse_args()
    groups = defaultdict(list)
    for f in sorted(Path(args.runs).glob("*/metrics.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        cond = c["condition"] if c["condition"] == "fixed" else f"changing q{c['q_early']:g}->{c['q_late']:g}@{c['shift_epoch']}"
        key = (c["model"], cond, c["weight_decay"], c["epochs"]) + ((c["split_seed"],) if args.by_split else ())
        groups[key].append(r)
    if not groups:
        raise SystemExit(f"no metrics.json under {args.runs}")
    rows = []
    for key, rs in sorted(groups.items()):
        row = {"model": key[0], "condition": key[1], "wd": key[2], "epochs": key[3], "n_runs": len(rs)}
        if args.by_split:
            row["split_seed"] = key[4]
        for name, fn in COLS:
            v = np.array([fn(r) for r in rs if fn(r) is not None], dtype=float)
            row[name] = f"{v.mean():.4f} +- {v.std(ddof=1) if len(v) > 1 else 0:.4f}" if len(v) else "-"
        rows.append(row)
    cols = list(rows[0].keys())
    print("\n".join(" | ".join(f"{c}={r[c]}" for c in cols) for r in rows))
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
