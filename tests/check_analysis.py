"""
Numpy-only tests of the analysis harness with PLANTED structure (so the tools are shown to find what is there, and
not to find what is not). Run from the project root:   python tests/check_analysis.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import core, report  # noqa: E402
from data import WorldData  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        FAILS.append(name)


def main():
    data = WorldData(0, 0)
    ct = core.cell_table(data)
    names = ct["op_names"]
    pos = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(0)
    print("[bookkeeping]")
    check("cell table has 256 pairs x 32 states", len(ct["Y"]) == 8192 and len(ct["pairs"]) == 256)
    sets = data.evaluation_sets()
    check("split counts match WorldData.evaluation_sets",
          all(int((ct["split"] == s).sum()) == len(sets[s]["y"]) for s in sets))
    check("x column is the math state: re-derived targets match", all(
        data.state_to_token[data.compositions[ct["pairs"][k]][int(ct["x"][k * 32 + j])]] == ct["Y"][k * 32 + j]
        for k in (0, 37, 101, 255) for j in (0, 13, 31)))
    idx = core.context_index(ct)
    ok = True
    for _ in range(300):
        s, i, k, x = int(rng.integers(2)), int(rng.integers(16)), int(rng.integers(16)), int(rng.integers(32))
        c = idx[s, i, k, x]
        first, second = (i, k) if s == 0 else (k, i)
        ok &= (ct["op_i"][c] == first and ct["op_j"][c] == second and ct["x"][c] == x)
    check("context_index maps (slot, op, partner, state) to the right cell", ok)
    agr = core.op_agreement(data)
    check("task agreement: R=1.0, G=D=0.5, U<=3/32", agr[pos["R1"], pos["R2"]] == 1.0 and agr[pos["G1"], pos["G2"]] == .5
          and agr[pos["D1"], pos["D2"]] == .5 and agr[pos["U1"], pos["U2"]] <= 3 / 32)

    print("[behavioural trajectories: planted family transition times]")
    epochs = np.arange(0, 601, 10)
    fam = np.array([n[0] for n in names])
    planted = {("A", "A"): 100, ("B", "B"): 150}
    T = np.full(len(ct["Y"]), 250.0)
    for (a, b), t0 in planted.items():
        T[(fam[ct["op_i"]] == a) & (fam[ct["op_j"]] == b)] = t0
    T[ct["split"] == "train"] = 30.0
    p = 1 / (1 + np.exp(-(epochs[:, None] - T[None, :]) / 15.0))
    correct = rng.random(p.shape) < p
    masks = core.group_masks(ct)
    cur = core.curves(correct, masks)
    te = core.transition_epoch(epochs, cur["set/train"])
    check("train transition recovered near planted 30 (+~35)", te is not None and 30 <= te <= 90, f"({te})")
    for key, t0 in (("heldout_family/A->A", 100), ("heldout_family/B->B", 150)):
        if key in cur:
            te = core.transition_epoch(epochs, cur[key])
            check(f"{key} transition recovered near planted {t0}", te is not None and t0 <= te <= t0 + 70, f"({te})")
    te_a, te_b = (core.transition_epoch(epochs, cur[k]) for k in ("heldout_family/A->A", "heldout_family/B->B"))
    check("ordering A->A before B->B recovered", te_a is not None and te_b is not None and te_a < te_b)
    ce = core.crossing_epoch(epochs, cur["set/test"], 0.5)
    check("test curve crosses 0.5 near planted 250", ce is not None and 220 <= ce <= 280, f"({ce})")
    check("transition_epoch needs `hold` consecutive checkpoints",
          core.transition_epoch([0, 1, 2, 3], [0, .95, .2, .95], hold=2) is None)
    fs = core.functions_solved_curve(correct, ct, "test")
    check("functions_solved curve rises from ~0 to ~1", fs[0] < .05 and fs[-1] > .9, f"({fs[0]:.2f}->{fs[-1]:.2f})")

    print("[D_R: planted merge of R1/R2]")
    d = 24
    ctx = rng.normal(size=(32, d))
    e1, e2 = rng.normal(size=(16, d)), rng.normal(size=(16, d))
    e1[pos["R2"]], e2[pos["R2"]] = e1[pos["R1"]], e2[pos["R1"]]          # R1 and R2 are interchangeable
    A = ctx[ct["x"]] + .5 * e1[ct["op_i"]] + .5 * e2[ct["op_j"]]
    H = core.gather_contexts(A, idx)
    valid = core.valid_contexts(ct, idx, ("train",))
    D = core.operator_swap_distance(H, valid)
    Dm = np.nanmean(D, axis=0)
    check("D_R(i,i) = 0 and matrix symmetric", np.allclose(np.diag(Dm), 0) and np.allclose(Dm, Dm.T, equal_nan=True))
    check("planted: D_R(R1,R2) = 0 while D_R(U1,U2) > 0", Dm[pos["R1"], pos["R2"]] < 1e-9 and Dm[pos["U1"], pos["U2"]] > .1,
          f"({Dm[pos['R1'], pos['R2']]:.3g} vs {Dm[pos['U1'], pos['U2']]:.3f})")
    summ = core.relation_summary(Dm, names, agr)
    check("R1/R2 is the closest of all pairs (percentile 0), U1/U2 is not", summ["R"]["percentile"] == 0 and summ["U"]["percentile"] > .1)
    check("D_R is scale-free (x3 activations -> same D_R)",
          np.allclose(np.nanmean(core.operator_swap_distance(3 * H, valid), 0), Dm, equal_nan=True))
    Hs = core.operator_swap_distance(H, np.zeros_like(valid))
    check("no valid contexts -> NaN (not 0)", np.isnan(Hs[0, 0, 1]))
    A0 = ctx[ct["x"]] + .5 * rng.normal(size=(16, d))[ct["op_i"]] + .5 * rng.normal(size=(16, d))[ct["op_j"]]
    D0 = np.nanmean(core.operator_swap_distance(core.gather_contexts(A0, idx), valid), 0)
    s0 = core.relation_summary(D0, names, agr)
    check("negative control: with no planted merge R1/R2 is not special", s0["R"]["percentile"] > .02, f"(pct {s0['R']['percentile']:.2f})")

    print("[positive control: Z_32 structure in embeddings]")
    ang = 2 * np.pi * np.outer(np.arange(32), [1, 3, 5]) / 32
    E_circ = np.concatenate([np.cos(ang), np.sin(ang)], 1) + .05 * rng.normal(size=(32, 6))
    E_perm = E_circ[rng.permutation(32)]
    E_rand = rng.normal(size=(32, 16))
    f_circ, f_perm, f_rand = (core.circulant_fraction(e) for e in (E_circ, E_perm, E_rand))
    null = core.circulant_null(E_circ, 300)
    check("circular embeddings score ~1", f_circ > .95, f"({f_circ:.3f})")
    check("same embeddings in scrambled state order score near the null", f_perm < np.quantile(null, .99) + 1e-9 and f_perm < .2, f"({f_perm:.3f})")
    null_r = core.circulant_null(E_rand, 300)
    check("structureless embeddings score near 16/496 = 0.03 regardless of dimension",
          f_rand < .12 and null_r.mean() < .08 and core.circulant_null(rng.normal(size=(32, 64)), 100).mean() < .08,
          f"(random {f_rand:.3f}, null mean d=16 {null_r.mean():.3f})")
    check("observed circular score exceeds null 99th percentile", f_circ > np.quantile(null, .99))

    print("[trajectory schema + report pipeline]")
    n_e = len(epochs)
    Dtraj = np.stack([np.stack([np.stack([D[0], D[1]])] * 3) for _ in range(n_e)])  # [n_e, 3 layers, 2, 16, 16]
    circ = np.linspace(.04, .9, n_e)
    traj = dict(epochs=epochs, correct=correct, logp=np.zeros(correct.shape, np.float32), split=ct["split"],
                op_i=ct["op_i"], op_j=ct["op_j"], x=ct["x"], function_id=ct["function_id"], op_names=np.array(names),
                agreement=agr, D=Dtraj.astype(np.float32), circ=circ, circ_null_mean=np.full(n_e, .05),
                circ_null_hi=np.full(n_e, .08))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trajectory.npz"
        core.save_trajectory(path, **traj)
        back = core.load_trajectory(path)
        check("trajectory round-trips through npz", np.array_equal(back["correct"], correct) and list(back["op_names"]) == names)
        try:
            core.save_trajectory(Path(tmp) / "bad.npz", epochs=epochs)
            check("incomplete trajectory is rejected", False)
        except AssertionError:
            check("incomplete trajectory is rejected", True)
        cur2, trans, rel_rows = report.build_tables(back)
        check("report tables built", len(trans) > 20 and len(rel_rows) == n_e * 3 * 4)
        report.make_figures(back, cur2, rel_rows, Path(tmp))
        pngs = sorted(p.name for p in Path(tmp).glob("fig_*.png"))
        check("four figures written", pngs == ["fig_circulant.png", "fig_dr_relations.png", "fig_families.png", "fig_sets.png"], str(pngs))
        rel_R = [r for r in rel_rows if r[2] == "R" and r[1] == 2]
        check("report recovers planted R1/R2 merge (percentile 0 at every epoch)", all(r[5] == 0 for r in rel_R))

    print(f"\n{len(FAILS)} failing check(s)" + (": " + "; ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
