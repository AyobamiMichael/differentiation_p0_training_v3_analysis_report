"""Torch-free helpers for run_p0.py (kept separate so they can be unit-tested without PyTorch)."""
import hashlib
import json
import math

import numpy as np


def lr_at(step, total_steps, lr, warmup_steps, min_lr_ratio=0.1):
    """Linear warmup then cosine decay to min_lr_ratio * lr."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def checkpoint_epochs(epochs, shift_epoch=None, every=0):
    """1-based 'after epoch e' checkpoint schedule. Default: dense log-spaced + linear + around the shift."""
    if every and every > 0:
        s = set(range(every, epochs + 1, every)) | {epochs}
    else:
        s = {int(round(v)) for v in np.geomspace(1, epochs, 30)}
        s |= {int(v) for v in np.linspace(0, epochs, 21)[1:]}
        if shift_epoch is not None:
            s |= {shift_epoch + d for d in (-2, -1, 0, 1, 2, 5, 10, 20)}
    return sorted(e for e in s if 1 <= e <= epochs)


def metrics_from_logits(logits, y, n_bins=10):
    """Accuracy plus probability-sensitive metrics. Returns (metrics dict, boolean correctness vector)."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    m = logits.max(1, keepdims=True)
    logp = logits - (m + np.log(np.exp(logits - m).sum(1, keepdims=True)))
    idx = np.arange(len(y))
    nll = -logp[idx, y]
    prob = np.exp(logp)
    pred, conf = prob.argmax(1), prob.max(1)
    correct = pred == y
    ece, edges = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.any():
            ece += mk.mean() * abs(correct[mk].mean() - conf[mk].mean())
    return {"accuracy": float(correct.mean()), "cross_entropy": float(nll.mean()),
            "correct_token_logp": float(-nll.mean()), "mean_prob_on_target": float(prob[idx, y].mean()),
            "mean_confidence": float(conf.mean()), "ece": float(ece),
            "confident_error_rate": float(((~correct) & (conf > 0.9)).mean()),
            "n": int(len(y))}, correct


def d_region_accuracy(correct, meta):
    """Accuracy on cells grouped by whether the D operand lies in the shared or divergent region."""
    out = {}
    for slot, is_d, sh in (("second", "j_is_d", "z_shared"), ("first", "i_is_d", "x_shared")):
        for region, mk in (("shared", meta[is_d] & meta[sh]), ("div", meta[is_d] & ~meta[sh])):
            out[f"d_{slot}_{region}"] = float(correct[mk].mean()) if mk.any() else None
            out[f"d_{slot}_{region}_n"] = int(mk.sum())
    return out


def merge_meta(*metas):
    return {k: np.concatenate([m[k] for m in metas]) for k in metas[0]}


def best_by_val(history, select="val", report="test", metric="accuracy"):
    """Checkpoint selection on VALIDATION only; reports the test value at the selected epoch."""
    rows = [r for r in history if select in r["eval"]]
    if not rows:
        return None
    best = max(rows, key=lambda r: (r["eval"][select][metric], -r["epoch"]))
    return {"epoch": best["epoch"], select: best["eval"][select][metric], report: best["eval"][report][metric]}


def first_epoch_ge(history, set_name, threshold, metric="accuracy"):
    for r in history:
        if set_name in r["eval"] and r["eval"][set_name][metric] >= threshold:
            return r["epoch"]
    return None


def jsonable(o):
    if isinstance(o, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return jsonable(o.item())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def config_hash(cfg):
    return hashlib.sha1(json.dumps(jsonable(cfg), sort_keys=True).encode()).hexdigest()[:12]
