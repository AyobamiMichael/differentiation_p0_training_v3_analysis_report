"""
Turn a trajectory.npz into tables and figures (numpy + matplotlib; no torch).

    python -m analysis.report --traj runs/<name>/analysis/trajectory.npz --out runs/<name>/analysis

Outputs: curves.csv, transitions.csv, dr_relations.csv, circulant.csv, fig_*.png, and a printed summary.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import core  # noqa: E402


def ct_from_traj(t):
    return {"split": t["split"], "op_i": t["op_i"], "op_j": t["op_j"], "x": t["x"],
            "function_id": t["function_id"], "op_names": [str(n) for n in t["op_names"]]}


def write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def build_tables(t, thr=0.9, hold=3):
    ct = ct_from_traj(t)
    epochs, correct = t["epochs"], t["correct"]
    cur = core.curves(correct, core.group_masks(ct))
    cur["functions_solved_test_90"] = core.functions_solved_curve(correct, ct, "test", 0.9)
    trans = []
    for name, c in cur.items():
        trans.append((name, core.transition_epoch(epochs, c, thr, hold), core.crossing_epoch(epochs, c, 0.5),
                      float(c[-1]), float(np.max(c)), int(epochs[int(np.argmax(c))])))
    names = ct["op_names"]
    rel_rows, layers = [], t["D"].shape[1]
    for e_i, e in enumerate(epochs):
        for l in range(layers):
            D = np.nanmean(t["D"][e_i, l], axis=0)
            summ = core.relation_summary(D, names, t["agreement"])
            for key in core.RELATIONS:
                r = summ[key]
                rel_rows.append((int(e), l, key, r["pair"], r["distance"], r["percentile"], r["agreement"],
                                 summ["all_pairs_median"]))
    return cur, trans, rel_rows


def make_figures(t, cur, rel_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = t["epochs"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in ("train", "id", "val", "test"):
        if f"set/{s}" in cur:
            ax.plot(epochs, cur[f"set/{s}"], label=s)
    ax.set_xlabel("epoch"), ax.set_ylabel("accuracy"), ax.legend(), ax.set_title("accuracy by evaluation set")
    fig.tight_layout(), fig.savefig(out / "fig_sets.png", dpi=130), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, c in cur.items():
        if k.startswith("heldout_family/"):
            ax.plot(epochs, c, label=k.split("/")[1])
    ax.set_xlabel("epoch"), ax.set_ylabel("held-out accuracy"), ax.legend(ncol=2, fontsize=7)
    ax.set_title("held-out accuracy by operator-family pair (first->second)")
    fig.tight_layout(), fig.savefig(out / "fig_families.png", dpi=130), plt.close(fig)

    last = max(r[1] for r in rel_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, color in zip(core.RELATIONS, ("C3", "C1", "C2", "C0")):
        rows = [r for r in rel_rows if r[2] == key and r[1] == last]
        axes[0].plot([r[0] for r in rows], [r[4] for r in rows], color=color, label=f"{key} {rows[0][3]}")
        axes[1].plot([r[0] for r in rows], [r[5] for r in rows], color=color, label=key)
    med = [r[7] for r in rel_rows if r[2] == "R" and r[1] == last]
    axes[0].plot([r[0] for r in rel_rows if r[2] == "R" and r[1] == last], med, "k--", label="median of all pairs")
    axes[0].set_title(f"D_R (operator-swap distance), final layer {last}"), axes[0].set_xlabel("epoch"), axes[0].legend(fontsize=7)
    axes[1].set_title("percentile among all 120 operator pairs (0 = closest)"), axes[1].set_xlabel("epoch"), axes[1].legend()
    fig.tight_layout(), fig.savefig(out / "fig_dr_relations.png", dpi=130), plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, t["circ"], label="observed")
    ax.plot(epochs, t["circ_null_mean"], "k--", label="permutation null mean")
    ax.plot(epochs, t["circ_null_hi"], "k:", label="permutation null 95th pct")
    ax.set_xlabel("epoch"), ax.set_ylabel("translation-invariant fraction"), ax.legend()
    ax.set_title("positive control: Z_32 structure in state embeddings")
    fig.tight_layout(), fig.savefig(out / "fig_circulant.png", dpi=130), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thr", type=float, default=0.9)
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    t = core.load_trajectory(args.traj)
    out = Path(args.out) if args.out else Path(args.traj).parent
    out.mkdir(parents=True, exist_ok=True)
    cur, trans, rel_rows = build_tables(t, args.thr, args.hold)
    write_csv(out / "curves.csv", ["epoch", "group", "accuracy"],
              [(int(e), k, float(c[i])) for k, c in cur.items() for i, e in enumerate(t["epochs"])])
    write_csv(out / "transitions.csv", ["group", f"first_epoch_ge_{args.thr}_for_{args.hold}_ckpts",
                                         "epoch_crossing_0.5", "final", "max", "epoch_of_max"], trans)
    write_csv(out / "dr_relations.csv", ["epoch", "layer", "relation", "pair", "D_R", "percentile", "task_agreement",
                                          "median_all_pairs"], rel_rows)
    write_csv(out / "circulant.csv", ["epoch", "observed", "null_mean", "null_95"],
              list(zip(t["epochs"].tolist(), t["circ"].tolist(), t["circ_null_mean"].tolist(), t["circ_null_hi"].tolist())))
    if not args.no_figures:
        make_figures(t, cur, rel_rows, out)

    print("\n=== transitions (first epoch with held-out/overall accuracy >= thr for `hold` consecutive checkpoints) ===")
    for row in trans:
        if row[0].startswith(("set/", "heldout_family/", "functions_solved")):
            print(f"  {row[0]:<34} transition epoch {row[1]}   crosses 50% at {row[2]}   final {row[3]:.3f}")
    last = max(r[1] for r in rel_rows)
    final_epoch = int(t["epochs"][-1])
    print(f"\n=== D_R at final layer ({last}), first vs last checkpoint (percentile 0 = closest of 120 pairs) ===")
    for e in (int(t["epochs"][0]), final_epoch):
        for r in (r for r in rel_rows if r[0] == e and r[1] == last):
            print(f"  epoch {e:>4} {r[2]} {r[3]:<7} D_R {r[4]:.3f}  percentile {r[5]:.2f}  task agreement {r[6]:.2f}")
    print(f"\n=== positive control: translation-invariant fraction of state-embedding geometry ===\n"
          f"  epoch {int(t['epochs'][0])}: {t['circ'][0]:.3f}   epoch {final_epoch}: {t['circ'][-1]:.3f}   "
          f"(permutation null 95th percentile at final epoch: {t['circ_null_hi'][-1]:.3f})")


if __name__ == "__main__":
    main()
