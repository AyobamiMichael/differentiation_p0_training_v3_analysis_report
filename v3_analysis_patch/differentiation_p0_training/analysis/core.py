"""
Numpy-only core of the checkpoint analysis harness (no torch import, fully unit-tested).

Cell convention: every (first operator, second operator, state) triple is a "cell"; cells are ordered pair-major with
pairs = sorted(data.compositions) and states ascending (cell = pair_idx * 32 + x, x = MATH state).
Each cell has a split label: train | id | val | test.

Contents
    cell_table / context_index / valid_contexts / gather_contexts   bookkeeping
    group_masks / curves / transition_epoch / crossing_epoch        behavioural trajectories
    operator_swap_distance / relation_summary                       D_R: representational differentiation
    circulant_fraction / circulant_null                             positive control: Z_32 structure in state embeddings
    save_trajectory / load_trajectory                               shared on-disk schema
"""
import numpy as np

N = 32
STATE0 = 16
RELATIONS = {"R": ("R1", "R2"), "G": ("G1", "G2"), "D": ("D1", "D2"), "U": ("U1", "U2")}


# ------------------------------------------------------------------------------------------------ bookkeeping
def cell_table(data):
    pairs = sorted(data.compositions)
    X, Y = data.all_examples(pairs)
    names = list(data.op_names)
    tok2state = np.argsort(data.state_to_token)
    x_math = tok2state[X[:, 2] - STATE0]
    pair_idx = np.repeat(np.arange(len(pairs)), N)
    split = np.empty(len(X), dtype="<U5")
    val, test = set(data.val_pairs), set(data.test_pairs)
    sig_ids, func_of_pair = {}, []
    for k, p in enumerate(pairs):
        sl = slice(k * N, (k + 1) * N)
        if p in test:
            split[sl] = "test"
        elif p in val:
            split[sl] = "val"
        else:
            held = np.asarray(data.id_cells[p], dtype=int)
            split[sl] = np.where(np.isin(x_math[sl], held), "id", "train")
        func_of_pair.append(sig_ids.setdefault(data.compositions[p], len(sig_ids)))
    return {"pairs": pairs, "X": X, "Y": Y, "op_i": X[:, 0].copy(), "op_j": X[:, 1].copy(), "x": x_math,
            "pair_idx": pair_idx, "split": split, "op_names": names,
            "function_id": np.repeat(np.asarray(func_of_pair), N)}


def op_agreement(data):
    """A[i,j] = mean_x [f_i(x) == f_j(x)]: task-level ground truth for how much two operators overlap."""
    arr = np.stack([data._arr[n] for n in data.op_names])
    return (arr[:, None, :] == arr[None, :, :]).mean(2)


def context_index(ct):
    """idx[s,i,k,x] = cell index of input (OP_i, OP_k, x) for s=0 (operator i FIRST) and (OP_k, OP_i, x) for s=1."""
    names = ct["op_names"]
    pair_pos = {p: n for n, p in enumerate(ct["pairs"])}
    n_op = len(names)
    idx = np.zeros((2, n_op, n_op, N), dtype=np.int64)
    for i, a in enumerate(names):
        for k, b in enumerate(names):
            idx[0, i, k] = pair_pos[(a, b)] * N + np.arange(N)
            idx[1, i, k] = pair_pos[(b, a)] * N + np.arange(N)
    return idx


def valid_contexts(ct, idx, which=("train",)):
    return np.isin(ct["split"][idx], list(which))


def gather_contexts(A, idx):
    """A: [n_cells, d] activations -> H: [2, n_op, n_op, 32, d]."""
    return A[idx]


# ------------------------------------------------------------------------------------------------ behaviour
def group_masks(ct):
    masks = {f"set/{s}": ct["split"] == s for s in ("train", "id", "val", "test") if (ct["split"] == s).any()}
    names = ct["op_names"]
    held = np.isin(ct["split"], ["val", "test"])  # held-out FUNCTIONS only
    fam = np.array([n[0] for n in names])
    for o, name in enumerate(names):
        masks[f"heldout_first_op/{name}"] = held & (ct["op_i"] == o)
        masks[f"heldout_second_op/{name}"] = held & (ct["op_j"] == o)
    for a in sorted(set(fam)):
        for b in sorted(set(fam)):
            m = held & (fam[ct["op_i"]] == a) & (fam[ct["op_j"]] == b)
            if m.any():
                masks[f"heldout_family/{a}->{b}"] = m
    return masks


def curves(correct, masks):
    return {k: correct[:, m].mean(1) for k, m in masks.items() if m.any()}


def functions_solved_curve(correct, ct, split="test", thr=0.9):
    m = ct["split"] == split
    f = ct["function_id"][m]
    n = np.bincount(f)
    keep = n > 0
    out = []
    for row in correct[:, m]:
        acc = np.bincount(f, weights=row.astype(float), minlength=len(n))[keep] / n[keep]
        out.append(float((acc >= thr).mean()))
    return np.asarray(out)


def transition_epoch(epochs, curve, thr=0.9, hold=3):
    """First epoch from which the curve stays >= thr for `hold` consecutive checkpoints (None if never)."""
    curve = np.asarray(curve)
    for i in range(len(curve) - hold + 1):
        if (curve[i:i + hold] >= thr).all():
            return int(epochs[i])
    return None


def crossing_epoch(epochs, curve, level):
    """First linearly-interpolated epoch at which the curve crosses `level` upward (None if never)."""
    curve = np.asarray(curve, dtype=float)
    for i in range(1, len(curve)):
        if curve[i - 1] < level <= curve[i]:
            f = (level - curve[i - 1]) / (curve[i] - curve[i - 1])
            return float(epochs[i - 1] + f * (epochs[i] - epochs[i - 1]))
    return None


# ------------------------------------------------------------------------------------------------ D_R
def operator_swap_distance(H, valid=None, min_contexts=30, spread_sample=300, seed=0):
    """
    Normalised operator-swap distance. H: [2, n_op, n_ctx_a, n_ctx_b, d]; contexts = all (partner, state) combinations.
    D[s,i,j] = mean over matched contexts c of ||h(i,c) - h(j,c)||  /  spread,
    spread   = mean distance between representations of the SAME operator in DIFFERENT contexts.
    D ~ 0: swapping the operator barely moves the QUERY representation; D ~ 1: swapping it matters as much as changing
    the whole context. valid[s,i,a,b] optionally restricts contexts; pairs with < min_contexts shared contexts -> NaN.
    """
    S, O = H.shape[:2]
    d = H.shape[-1]
    Hf = H.reshape(S, O, -1, d).astype(np.float64)
    vf = np.ones(Hf.shape[:3], dtype=bool) if valid is None else valid.reshape(S, O, -1)
    rng = np.random.default_rng(seed)
    spreads = []
    for s in range(S):
        for o in range(O):
            ctx = Hf[s, o][vf[s, o]]
            if len(ctx) < 2:
                continue
            ctx = ctx[rng.permutation(len(ctx))[:spread_sample]]
            dist = np.linalg.norm(ctx[:, None, :] - ctx[None, :, :], axis=2)
            spreads.append(dist.sum() / (len(ctx) * (len(ctx) - 1)))
    spread = float(np.mean(spreads)) if spreads else np.nan
    if not np.isfinite(spread) or spread < 1e-12:
        # e.g. the embedding site at the QUERY position: identical for every input, so D_R is undefined (not 0/0 noise)
        D = np.full((S, O, O), np.nan)
        for s in range(S):
            np.fill_diagonal(D[s], 0.0)
        return D
    D = np.full((S, O, O), np.nan)
    for s in range(S):
        for i in range(O):
            D[s, i, i] = 0.0
            for j in range(i + 1, O):
                m = vf[s, i] & vf[s, j]
                if m.sum() >= min_contexts:
                    D[s, i, j] = D[s, j, i] = np.linalg.norm(Hf[s, i][m] - Hf[s, j][m], axis=1).mean() / spread
    return D


def relation_summary(D, names, agreement=None):
    """D: [n_op, n_op] (e.g. nanmean over the two slots). Returns, for each controlled pair, its distance and its
    percentile among all unordered operator pairs (0 = closest of all)."""
    n = len(names)
    iu = np.triu_indices(n, 1)
    all_d = D[iu]
    all_d = all_d[~np.isnan(all_d)]
    out = {"n_pairs": int(len(all_d)), "all_pairs_mean": float(all_d.mean()), "all_pairs_median": float(np.median(all_d))}
    pos = {nm: k for k, nm in enumerate(names)}
    for key, (a, b) in RELATIONS.items():
        v = D[pos[a], pos[b]]
        out[key] = {"pair": f"{a}|{b}", "distance": float(v),
                    "percentile": float((all_d < v).mean()) if not np.isnan(v) else None,
                    "agreement": float(agreement[pos[a], pos[b]]) if agreement is not None else None}
    return out


# ------------------------------------------------------------------------------------------------ positive control
def circulant_fraction(E):
    """E: [32, d] state embeddings ordered by MATH state. Fraction of the variance of the OFF-DIAGONAL similarities
    (centred Gram matrix) that is explained by the lag (x - y) mod 32, i.e. translation-invariant structure on Z_32.
    Two trivially circulant components are removed so the structureless baseline (~ 15/495 = 0.03) does not depend on the
    embedding dimension: the diagonal (self-similarity) and the global mean of the off-diagonal (for d > 31 centred
    random embeddings have a constant negative similarity). 1 = pure Fourier-type structure."""
    E = np.asarray(E, dtype=np.float64)
    E = E - E.mean(0, keepdims=True)
    G = E @ E.T
    n = len(G)
    lag = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    off = lag != 0
    mu = G[off].mean()
    explained = sum(n * (G[lag == k].mean() - mu) ** 2 for k in range(1, n))
    total = ((G[off] - mu) ** 2).sum()
    return float(explained / total)


def circulant_null(E, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([circulant_fraction(np.asarray(E)[rng.permutation(len(E))]) for _ in range(n)])


# ------------------------------------------------------------------------------------------------ on-disk schema
SCHEMA = ("epochs", "correct", "logp", "split", "op_i", "op_j", "x", "function_id", "op_names", "agreement",
          "D", "circ", "circ_null_mean", "circ_null_hi")


def save_trajectory(path, **arrays):
    missing = [k for k in SCHEMA if k not in arrays]
    assert not missing, f"missing trajectory arrays: {missing}"
    np.savez_compressed(path, **arrays)


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
