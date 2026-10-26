"""
Collect a trajectory for one run (torch layer; everything numerical lives in analysis/core.py).

    python -m analysis.collect --run runs/task0_split0_train3_m1_fixed_wd0.3
    python -m analysis.collect --run runs/<name> --every 2 --contexts train --out runs/<name>/analysis

Writes <out>/trajectory.npz (schema: core.SCHEMA) and <out>/reproduction_check.json.

Step 1 (gate): for every checkpoint whose epoch was logged in metrics.json, recompute train/id/val/test accuracy and
compare with the logged value. If this check fails, do not use the trajectory.
Epoch 0 is a freshly initialised model rebuilt with the training seed (seed_all(train_seed) then model construction,
the same order as run_p0.train_one). Its accuracy should be near chance; this is a sanity check on that assumption.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import core  # noqa: E402
from data import WorldData  # noqa: E402
from models import MODEL_CONFIGS, TinyTransformer  # noqa: E402
from run_p0 import seed_all  # noqa: E402


def load_run(run_dir):
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text())
    cfg = m["config"]
    data = WorldData(cfg["task_seed"], cfg["split_seed"], cfg["test_fraction"], cfg["val_fraction"],
                     cfg["id_holdout"], cfg["min_slot_coverage"])
    if data.fingerprint() != cfg["data_fingerprint"]:
        raise SystemExit(f"data fingerprint mismatch: rebuilt {data.fingerprint()} vs logged {cfg['data_fingerprint']}")
    ckpts = sorted(run_dir.glob("checkpoint_epoch_*.pt"))
    if not ckpts:
        raise SystemExit(f"no checkpoint_epoch_*.pt in {run_dir} (rerun training with --save-checkpoints)")
    return m, cfg, data, ckpts


def load_model(path, device):
    state = torch.load(path, map_location=device)
    model = TinyTransformer(**state["config"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval(), int(state["epoch"])


def init_model(cfg, device):
    seed_all(cfg["train_seed"])
    return TinyTransformer(**MODEL_CONFIGS[cfg["model"]]).to(device).eval()


@torch.no_grad()
def run_model(model, tokens, device, batch_size=4096):
    """Returns logits [n, 32] and QUERY-position residual states per site: site 0 = embeddings, site l = layer l output."""
    model.eval()
    n_sites = len(model.encoder.layers) + 1
    states, logits = [[] for _ in range(n_sites)], []
    for s in range(0, len(tokens), batch_size):
        xb = torch.as_tensor(tokens[s:s + batch_size], dtype=torch.long, device=device)
        caps = []
        hooks = [layer.register_forward_hook(lambda mod, inp, out, caps=caps: caps.append(out[:, -1, :].detach().float().cpu()))
                 for layer in model.encoder.layers]
        pos = torch.arange(xb.shape[1], device=device).unsqueeze(0)
        states[0].append((model.token_emb(xb) + model.pos_emb(pos))[:, -1, :].detach().float().cpu())
        out = model(xb)
        for h in hooks:
            h.remove()
        for l, c in enumerate(caps):
            states[l + 1].append(c)
        logits.append(out.float().cpu())
    return torch.cat(logits).numpy(), [torch.cat(s).numpy() for s in states]


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
    ap.add_argument("--contexts", choices=["train", "all"], default="train",
                    help="cells used as matched contexts for D_R")
    ap.add_argument("--no-activations", action="store_true")
    ap.add_argument("--null-samples", type=int, default=500)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    metrics, cfg, data, ckpts = load_run(run_dir)
    ct = core.cell_table(data)
    idx = core.context_index(ct)
    valid = core.valid_contexts(ct, idx, ("train",) if args.contexts == "train" else ("train", "id", "val", "test"))
    logged = {r["epoch"]: r["eval"] for r in metrics["history"]}
    s2t = data.state_to_token

    items = [(0, init_model(cfg, device))]
    for path in ckpts[::max(1, args.every)]:
        model, epoch = load_model(path, device)
        items.append((epoch, model))
    epochs, correct, logp, D_all, circ, null_mean, null_hi, checks = [], [], [], [], [], [], [], []
    for epoch, model in items:
        logits, states = run_model(model, ct["X"], device)
        c = logits.argmax(1) == ct["Y"]
        lp = log_softmax(logits.astype(np.float64))[np.arange(len(ct["Y"])), ct["Y"]]
        epochs.append(epoch), correct.append(c), logp.append(lp.astype(np.float32))
        if epoch in logged:  # reproduction gate
            for name in ("train", "id", "val", "test"):
                m = ct["split"] == name
                if name in logged[epoch] and m.any():
                    checks.append({"epoch": epoch, "set": name, "recomputed": float(c[m].mean()),
                                   "logged": logged[epoch][name]["accuracy"],
                                   "abs_diff": abs(float(c[m].mean()) - logged[epoch][name]["accuracy"])})
        if args.no_activations:
            D_all.append(np.full((len(states), 2, 16, 16), np.nan))
        else:
            D_all.append(np.stack([core.operator_swap_distance(core.gather_contexts(A, idx), valid) for A in states]))
        E = model.token_emb.weight.detach().float().cpu().numpy()[core.STATE0 + s2t]  # row x = embedding of MATH state x
        circ.append(core.circulant_fraction(E))
        null = core.circulant_null(E, args.null_samples)
        null_mean.append(null.mean()), null_hi.append(np.quantile(null, 0.95))
        print(f"epoch {epoch:>4}: train {c[ct['split'] == 'train'].mean():.3f} test {c[ct['split'] == 'test'].mean():.3f} "
              f"circulant {circ[-1]:.3f} (null95 {null_hi[-1]:.3f})", flush=True)

    worst = max((c["abs_diff"] for c in checks), default=None)
    init_acc = float(correct[0][ct["split"] == "train"].mean())
    report = {"n_checked": len(checks), "max_abs_diff": worst, "passed": worst is not None and worst < 1e-3,
              "init_train_accuracy": init_acc, "init_near_chance": init_acc < 0.08, "details": checks}
    (out / "reproduction_check.json").write_text(json.dumps(report, indent=2))
    print(f"\nREPRODUCTION GATE: {'PASS' if report['passed'] else 'FAIL'} "
          f"(max |recomputed - logged| = {worst}; epoch-0 train acc {init_acc:.3f})")
    core.save_trajectory(out / "trajectory.npz", epochs=np.array(epochs), correct=np.stack(correct),
                         logp=np.stack(logp), split=ct["split"], op_i=ct["op_i"], op_j=ct["op_j"], x=ct["x"],
                         function_id=ct["function_id"], op_names=np.array(ct["op_names"]),
                         agreement=core.op_agreement(data), D=np.stack(D_all), circ=np.array(circ),
                         circ_null_mean=np.array(null_mean), circ_null_hi=np.array(null_hi))
    print(f"wrote {out / 'trajectory.npz'}")


if __name__ == "__main__":
    main()
