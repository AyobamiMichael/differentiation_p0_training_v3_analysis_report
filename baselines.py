"""
Baselines and reference solvers (numpy only).

Entry points
    evaluate_all_baselines(data, seed)          -> {"val": {...}, "test": {...}}
    evaluate_baselines(data, train_pairs, eval_pairs, seed)   (v1-compatible signature)
    breakdown(data, pairs, correct)             -> per-pair / per-family / per-unique-function accuracy

Reference points reported for each evaluation set (all use TRAINING TUPLES ONLY, never the true operators):
    random                         chance
    triple_lookup                  (i,j,x)->y table: coverage of the eval set (0 by construction) + fallback chance
    composition_lookup_by_pair_label / _by_signature
                                   coverage only; by-signature coverage must be 0 (leak check)
    shortcut_accuracy              argmax P(y|x), P(y|i,x), P(y|j,x) estimated from the training cells
    rectangle_solver_full_tables   F_ij = F_i'j o F_i'j'^-1 o F_ij' using COMPLETE tables of the trained pairs
                                   (identifiability ceiling; more information than the network sees)
    rectangle_solver_matched_cells same solver restricted to the cells the network actually trains on
                                   (information-matched reference; withheld ID states are unavailable to it)
    generator_selfcheck            labels produced by the data module agree with an independent code path
"""
from collections import Counter, defaultdict

import numpy as np

N = 32
STATE0 = 16  # STATE_TOKEN_START in data.py


def _tables(data, pairs, cells=None):
    """pair -> length-32 array T with T[x_token_index] = y_token_index (-1 where the cell is unavailable).
    Built from example tuples (token space) only."""
    tables = {}
    for pair in pairs:
        xs = np.arange(N) if cells is None else np.asarray(cells[pair], dtype=np.int64)
        tok, y, _, _ = data._pair_block(pair, xs)
        t = np.full(N, -1, dtype=int)
        t[tok[:, 2] - STATE0] = y
        tables[pair] = t
    return tables


def _inverse(t):
    inv = np.full(N, -1, dtype=int)
    for x, y in enumerate(t):
        if y >= 0:
            inv[y] = x
    return inv


def generator_selfcheck(data, pairs):
    """Labels from the example builder vs data.compositions (built by world.operators.compose)."""
    X, Y = data.all_examples(pairs)
    for k, pair in enumerate(pairs):
        sig = data.compositions[pair]
        for x in range(N):
            tok_x = X[k * N + x, 2] - STATE0
            math_x = int(np.flatnonzero(data.state_to_token == tok_x)[0])
            if Y[k * N + x] != data.state_to_token[sig[math_x]]:
                return False
    return True


def rectangle_predict(tables, i, j, names):
    """Predict F_ij from trained pairs only. Returns (pred[32] with -1 where undetermined, n_rectangles)."""
    votes = [Counter() for _ in range(N)]
    n_rect = 0
    for ip in names:
        if (ip, j) not in tables:
            continue
        for jp in names:
            if (i, jp) in tables and (ip, jp) in tables:
                n_rect += 1
                inv = _inverse(tables[(ip, jp)])
                for x in range(N):
                    a = tables[(i, jp)][x]
                    b = inv[a] if a >= 0 else -1
                    c = tables[(ip, j)][b] if b >= 0 else -1
                    if c >= 0:
                        votes[x][int(c)] += 1
    pred = np.array([v.most_common(1)[0][0] if v else -1 for v in votes])
    return pred, n_rect


def _solver_accuracy(tables, eval_tab, pairs, names, rng):
    correct, solved, rect = 0, 0, []
    for p in pairs:
        pred, n_rect = rectangle_predict(tables, p[0], p[1], names)
        rect.append(n_rect)
        solved += int((pred >= 0).all())
        pred = np.where(pred >= 0, pred, rng.integers(0, N, size=N))  # undetermined cells: random guess
        correct += int((pred == eval_tab[p]).sum())
    return {"accuracy": correct / (len(pairs) * N), "pairs_fully_determined": solved / len(pairs),
            "min_rectangles_per_pair": int(min(rect))}


def evaluate_baselines(data, train_pairs, eval_pairs, seed=0):
    rng = np.random.default_rng(seed)
    names = data.op_names
    eval_tab = _tables(data, eval_pairs)
    n_eval = len(eval_pairs) * N
    cells = getattr(data, "train_cells", None)
    full_tab = _tables(data, train_pairs)
    matched_tab = _tables(data, train_pairs, cells) if cells is not None else full_tab

    random_acc = float(np.mean([(rng.integers(0, N, size=N) == eval_tab[p]).mean() for p in eval_pairs]))
    train_cells = {p: set(map(int, cells[p])) if cells is not None else set(range(N)) for p in train_pairs}
    triple_cov = float(np.mean([x in train_cells[p] if p in train_cells else False
                                for p in eval_pairs for x in range(N)]))
    label_set = set(train_pairs)
    label_cov = float(np.mean([p in label_set for p in eval_pairs]))
    train_sigs = {data.compositions[p] for p in train_pairs}
    sig_cov = float(np.mean([data.compositions[p] in train_sigs for p in eval_pairs]))

    def probe(keyfn):
        cnt = defaultdict(Counter)
        for (i, j), t in matched_tab.items():
            for x in range(N):
                if t[x] >= 0:
                    cnt[keyfn(i, j, x)][int(t[x])] += 1
        hit = 0
        for (i, j), t in eval_tab.items():
            for x in range(N):
                c = cnt.get(keyfn(i, j, x))
                hit += int(bool(c) and c.most_common(1)[0][0] == t[x])
        return hit / n_eval

    return {
        "n_eval_pairs": len(eval_pairs), "chance": 1.0 / N, "random_accuracy": random_acc,
        "triple_lookup": {"coverage": triple_cov, "fallback_accuracy": 1.0 / N},
        "composition_lookup_by_pair_label": {"coverage": label_cov, "fallback_accuracy": 1.0 / N},
        "composition_lookup_by_signature": {"coverage": sig_cov, "must_be_zero": True},
        "shortcut_accuracy": {"P(y|x)": probe(lambda i, j, x: x), "P(y|i,x)": probe(lambda i, j, x: (i, x)),
                              "P(y|j,x)": probe(lambda i, j, x: (j, x))},
        "rectangle_solver_full_tables": _solver_accuracy(full_tab, eval_tab, eval_pairs, names, rng),
        "rectangle_solver_matched_cells": _solver_accuracy(matched_tab, eval_tab, eval_pairs, names, rng),
        "generator_selfcheck": generator_selfcheck(data, eval_pairs),
    }


def evaluate_all_baselines(data, seed=0):
    return {"val": evaluate_baselines(data, data.train_pairs, data.val_pairs, seed),
            "test": evaluate_baselines(data, data.train_pairs, data.test_pairs, seed),
            "train_generator_selfcheck": generator_selfcheck(data, data.train_pairs)}


def breakdown(data, pairs, correct):
    """correct: boolean vector aligned with data.all_examples(pairs) (pair-major, state ascending)."""
    correct = np.asarray(correct, dtype=float).reshape(len(pairs), N)
    per_pair = {p: float(a) for p, a in zip(pairs, correct.mean(1))}
    fam = lambda n: n[0]
    by_family, by_function = defaultdict(list), defaultdict(list)
    for p, a in per_pair.items():
        by_family[(fam(p[0]), fam(p[1]))].append(a)
        by_function[data.compositions[p]].append(a)
    func_acc = [float(np.mean(v)) for v in by_function.values()]
    return {
        "pair_mean": float(np.mean(list(per_pair.values()))),
        "unique_function_mean": float(np.mean(func_acc)),
        "n_pairs": len(per_pair), "n_unique_functions": len(func_acc),
        "functions_solved_ge_90pct": int(sum(a >= 0.9 for a in func_acc)),
        "functions_near_chance_le_10pct": int(sum(a <= 0.1 for a in func_acc)),
        "by_family": {f"{a}->{b}": {"mean": float(np.mean(v)), "n_pairs": len(v)}
                      for (a, b), v in sorted(by_family.items())},
        "per_pair": per_pair,
    }
