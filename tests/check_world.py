"""
Numpy-only checks (no torch). Run from the project root:   python tests/check_world.py [--seeds 0 1 2]
Exits non-zero if any check fails.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import breakdown, evaluate_all_baselines  # noqa: E402
from data import WorldData  # noqa: E402
from utils import (best_by_val, checkpoint_epochs, d_region_accuracy, first_epoch_ge, jsonable, lr_at,  # noqa: E402
                   merge_meta, metrics_from_logits)

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        FAILS.append(name)


def world_checks(ts, ss):
    d = WorldData(ts, ss)
    print(f"\n[task_seed={ts} split_seed={ss}] {d.split_report}")
    ops = d.ops
    agree = lambda a, b: sum(x == y for x, y in zip(ops[a].mapping, ops[b].mapping))
    check("all operators are bijections", all(sorted(o.mapping) == list(range(32)) for o in ops.values()))
    check("R1 == R2 on 32/32", agree("R1", "R2") == 32)
    check("G1/G2 agree on exactly 16", agree("G1", "G2") == 16)
    check("D1/D2 agree on exactly 16 and shared set is the equality set",
          agree("D1", "D2") == 16 and {x for x in range(32) if ops["D1"](x) == ops["D2"](x)} == set(d.d_shared))
    check("U1/U2 overlap <= 3", agree("U1", "U2") <= 3, f"({agree('U1', 'U2')})")

    sig = d.compositions
    s = lambda ps: {sig[p] for p in ps}
    check("train/val/test composite functions are pairwise disjoint",
          not (s(d.train_pairs) & s(d.val_pairs)) and not (s(d.train_pairs) & s(d.test_pairs))
          and not (s(d.val_pairs) & s(d.test_pairs)))
    check("per-slot coverage >= 8 in training",
          d.split_report["min_train_pairs_as_first"] >= 8 and d.split_report["min_train_pairs_as_second"] >= 8)

    sets = d.evaluation_sets()
    seen = {(sig[p], int(x)) for p, xs in d.train_cells.items() for x in xs}
    leak = sum((sig[p], int(x)) in seen for p, xs in d.id_cells.items() for x in xs)
    check("ID cells: no (function, state) cell also appears in training", leak == 0, f"(leaks={leak})")
    same_class = {}
    ok = True
    for p in d.train_pairs:
        h = tuple(d.holdout_by_signature[sig[p]])
        ok &= tuple(d.id_cells[p]) == h
        same_class.setdefault(sig[p], set()).add(tuple(d.id_cells[p]))
    check("functionally equal pairs share the same held-out states", ok and all(len(v) == 1 for v in same_class.values()))
    check("evaluation sets have expected sizes",
          len(sets["val"]["y"]) == 32 * len(d.val_pairs) and len(sets["test"]["y"]) == 32 * len(d.test_pairs)
          and len(sets["train"]["y"]) + len(sets["id"]["y"]) == 32 * len(d.train_pairs))

    # sampler: budgets and exposures
    n_fixed = len(d.training_examples(0, "fixed")[0])
    Xf, Yf, inf_f = d.training_examples(0, "fixed")
    Xe, Ye, inf_e = d.training_examples(0, "changing", .9, .1, 30, 0, seed=1)
    Xl, Yl, inf_l = d.training_examples(40, "changing", .9, .1, 30, 0, seed=1)
    X5, Y5, _ = d.training_examples(0, "changing", .5, .5, 30, 0, seed=1)
    check("matched budget: examples/epoch equal in fixed, early and late", n_fixed == len(Xe) == len(Xl),
          f"({n_fixed}/{len(Xe)}/{len(Xl)})")
    check("operator marginals identical across conditions",
          all((np.bincount(a[:, k], minlength=16) == np.bincount(Xf[:, k], minlength=16)).all()
              for a in (Xe, Xl) for k in (0, 1)))
    check("early D-as-second exposure ~ 0.9", abs(inf_e["d_second_exposure"] - .9) < .04, f"({inf_e['d_second_exposure']:.3f})")
    check("late D-as-second exposure ~ 0.1", abs(inf_l["d_second_exposure"] - .1) < .04, f"({inf_l['d_second_exposure']:.3f})")
    check("early D-as-FIRST exposure ~ 0.9 (now controlled)", abs(inf_e["d_first_exposure"] - .9) < .06, f"({inf_e['d_first_exposure']:.3f})")
    check("late D-as-FIRST exposure ~ 0.1", abs(inf_l["d_first_exposure"] - .1) < .06, f"({inf_l['d_first_exposure']:.3f})")
    check("fixed exposure is the unmodified table (~0.5)", abs(inf_f["d_second_exposure"] - .5) < .06,
          f"({inf_f['d_second_exposure']:.3f})")
    same = (sorted(map(tuple, np.c_[Xf, Yf].tolist())) == sorted(map(tuple, np.c_[X5, Y5].tolist())))
    check("changing(q=0.5/0.5) is exactly the fixed table (no-shift control)", same)
    tok2state = np.argsort(d.state_to_token)
    id_set = {(p[0], p[1], int(x)) for p, xs in d.id_cells.items() for x in xs}
    emitted = {(d.op_names[int(r[0])], d.op_names[int(r[1])], int(tok2state[int(r[2]) - 16])) for r in Xe}
    check("sampler (changing) emits no held-out (pair, state) cell", not (emitted & id_set))

    # baselines
    b = evaluate_all_baselines(d, 0)
    t = b["test"]
    check("generator self-check passes", t["generator_selfcheck"] and b["train_generator_selfcheck"])
    check("signature-keyed lookup has zero coverage on test and val",
          t["composition_lookup_by_signature"]["coverage"] == 0 and b["val"]["composition_lookup_by_signature"]["coverage"] == 0)
    check("shortcut predictors near chance (< 0.08)", max(t["shortcut_accuracy"].values()) < .08,
          f"({max(t['shortcut_accuracy'].values()):.3f})")
    check("rectangle solver (full tables) solves test", t["rectangle_solver_full_tables"]["accuracy"] > .99,
          f"({t['rectangle_solver_full_tables']['accuracy']:.3f})")
    check("rectangle solver (matched cells) solves most of test", t["rectangle_solver_matched_cells"]["accuracy"] > .9,
          f"({t['rectangle_solver_matched_cells']['accuracy']:.3f})")

    # breakdown with a perfect predictor and a random one
    bd = breakdown(d, d.test_pairs, np.ones(32 * len(d.test_pairs), bool))
    check("breakdown: perfect predictor -> 1.0 at pair and function level",
          bd["pair_mean"] == 1.0 and bd["unique_function_mean"] == 1.0 and bd["n_unique_functions"] == d.split_report["test_unique_functions"])
    return d


def util_checks():
    print("\n[utils]")
    lr = [lr_at(s, 1000, 1e-3, 100, .1) for s in range(1000)]
    check("lr: warmup rises, peak ~1e-3, ends at ~1e-4", lr[0] < lr[50] < lr[99] and abs(max(lr) - 1e-3) < 1e-6 and abs(lr[-1] - 1e-4) < 2e-6)
    ce = checkpoint_epochs(600, 300)
    check("checkpoints: sorted, include 1, 600 and the shift", ce == sorted(set(ce)) and 1 in ce and 600 in ce and 300 in ce and len(ce) > 30, f"({len(ce)} ckpts)")
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 32))
    y = logits.argmax(1)
    m, c = metrics_from_logits(logits, y)
    check("metrics: argmax labels -> accuracy 1.0, finite CE", m["accuracy"] == 1.0 and np.isfinite(m["cross_entropy"]))
    u, _ = metrics_from_logits(np.zeros((100, 32)), np.zeros(100, int))
    check("metrics: uniform logits -> CE = ln 32", abs(u["cross_entropy"] - np.log(32)) < 1e-9)
    conf_wrong = np.zeros((10, 32))
    conf_wrong[:, 0] = 20
    w, _ = metrics_from_logits(conf_wrong, np.ones(10, int))
    check("metrics: confident-wrong predictions give CE > ln 32 and confident_error_rate 1", w["cross_entropy"] > np.log(32) and w["confident_error_rate"] == 1.0)
    hist = [{"epoch": e, "eval": {"val": {"accuracy": a}, "test": {"accuracy": a / 2}}} for e, a in [(1, .1), (5, .6), (10, .5)]]
    check("selection uses validation: picks epoch 5, reports test at 5", best_by_val(hist) == {"epoch": 5, "val": .6, "test": .3})
    check("first_epoch_ge", first_epoch_ge(hist, "val", .5) == 5 and first_epoch_ge(hist, "val", .9) is None)
    check("jsonable: tuple keys and numpy scalars", jsonable({("A1", "B1"): np.float64(1.5), "x": np.arange(2)}) == {"A1|B1": 1.5, "x": [0, 1]})
    meta = {"i_is_d": np.array([0, 0, 1], bool), "j_is_d": np.array([1, 1, 0], bool),
            "x_shared": np.array([1, 0, 1], bool), "z_shared": np.array([1, 0, 0], bool)}
    r = d_region_accuracy(np.array([True, False, True]), meta)
    check("d_region_accuracy", r["d_second_shared"] == 1.0 and r["d_second_div"] == 0.0 and r["d_first_shared"] == 1.0)
    check("merge_meta", len(merge_meta(meta, meta)["j_is_d"]) == 6)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    for s in args.seeds:
        world_checks(s, 0)
    world_checks(0, 1)
    util_checks()
    print(f"\n{len(FAILS)} failing check(s)" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
