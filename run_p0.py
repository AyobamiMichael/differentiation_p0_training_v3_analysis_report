"""
P0 runner (v2). Thin torch layer on top of the numpy-only data / baselines / utils modules.

Examples
    python run_p0.py --mode smoke
    python run_p0.py --mode train --model m1 --condition fixed --weight-decay 0.01 --epochs 600 --save-checkpoints
    python run_p0.py --mode sweep --estimate-only
    python run_p0.py --mode sweep --models m1 --weight-decay 0 0.01 --train-seed 0 1 2 --split-seed 0 --condition fixed
    python run_p0.py --mode train --condition changing --q-early 0.9 --q-late 0.1 --shift-epoch 300

Evaluation sets logged every --eval-every epochs: train (trained cells), id (held-out states of trained
functions), val (held-out functions; use this for ALL selection), test (held-out functions; report only).
"""
import argparse
import itertools
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from baselines import breakdown, evaluate_all_baselines
from data import WorldData
from models import MODEL_CONFIGS, TinyTransformer
from utils import (best_by_val, checkpoint_epochs, config_hash, d_region_accuracy, first_epoch_ge, jsonable,
                   lr_at, merge_meta, metrics_from_logits)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_logits(model, x, device, batch_size=4096):
    model.eval()
    outs = []
    for s in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[s:s + batch_size], dtype=torch.long, device=device)
        outs.append(model(xb).float().cpu())
    return torch.cat(outs).numpy()


def evaluate_sets(model, sets, device):
    res, correct = {}, {}
    for name, s in sets.items():
        res[name], correct[name] = metrics_from_logits(predict_logits(model, s["x"], device), s["y"])
    seen = [n for n in ("train", "id") if n in sets]
    res["d_regions"] = d_region_accuracy(np.concatenate([correct[n] for n in seen]),
                                         merge_meta(*[sets[n]["meta"] for n in seen]))
    return res, correct


def make_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]     # matrices and embeddings
    no_decay = [p for p in model.parameters() if p.ndim < 2]   # biases and LayerNorm
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999))


def run_name(model_name, task_seed, split_seed, train_seed, condition, wd, args, shift_epoch):
    cond = condition if condition == "fixed" else f"changing_q{args.q_early:g}-{args.q_late:g}_s{shift_epoch}"
    return f"task{task_seed}_split{split_seed}_train{train_seed}_{model_name}_{cond}_wd{wd:g}"


def train_one(args, model_name, task_seed, split_seed, train_seed, condition, wd):
    shift_epoch = args.shift_epoch if args.shift_epoch is not None else args.epochs // 2
    out = Path(args.outdir) / run_name(model_name, task_seed, split_seed, train_seed, condition, wd, args, shift_epoch)
    if args.skip_existing and (out / "metrics.json").exists():
        print(f"skip (exists): {out.name}")
        return None
    out.mkdir(parents=True, exist_ok=True)

    seed_all(train_seed)
    data = WorldData(task_seed, split_seed, args.test_fraction, args.val_fraction, args.id_holdout,
                     args.min_slot_coverage)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    sets = data.evaluation_sets()
    baselines = evaluate_all_baselines(data, train_seed)
    model = TinyTransformer(**MODEL_CONFIGS[model_name]).to(device)
    opt = make_optimizer(model, args.lr, wd)

    data_seed = train_seed + task_seed * 10_000
    sampler = dict(condition=condition, q_early=args.q_early, q_late=args.q_late,
                   shift_epoch=shift_epoch, ramp_epochs=args.ramp_epochs, seed=data_seed)
    n_epoch = len(data.training_examples(0, **sampler)[0])
    steps_per_epoch = math.ceil(n_epoch / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    ckpt_epochs = set(checkpoint_epochs(args.epochs, shift_epoch if condition == "changing" else None,
                                        args.checkpoint_every)) if args.save_checkpoints else set()
    cfg = dict(vars(args), model=model_name, task_seed=task_seed, split_seed=split_seed, train_seed=train_seed,
               condition=condition, weight_decay=wd, shift_epoch=shift_epoch, steps_per_epoch=steps_per_epoch,
               total_steps=total_steps, data_fingerprint=data.fingerprint())
    cfg_hash = config_hash({k: v for k, v in cfg.items() if k not in ("outdir", "skip_existing", "estimate_only")})
    print(f"\n=== {out.name} | {n_epoch} examples/epoch, {steps_per_epoch} steps/epoch, {total_steps} steps "
          f"| config {cfg_hash} | data {cfg['data_fingerprint']}", flush=True)

    history, step, t0 = [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        X, Y, info = data.training_examples(epoch - 1, **sampler)
        xt = torch.as_tensor(X, dtype=torch.long, device=device)
        yt = torch.as_tensor(Y, dtype=torch.long, device=device)
        perm = torch.randperm(len(xt), device=device)
        model.train()
        loss_sum, n_correct = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb, yb = xt[idx], yt[idx]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, args.warmup_steps, args.min_lr_ratio)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            loss_sum += loss.item() * len(idx)
            n_correct += int((logits.argmax(-1) == yb).sum())
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            res, _ = evaluate_sets(model, sets, device)
            history.append({"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"],
                            "train_epoch_loss": loss_sum / len(xt), "train_epoch_accuracy": n_correct / len(xt),
                            "exposure": info, "eval": res})
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                a = lambda n: res[n]["accuracy"] if n in res else float("nan")
                print(f"  ep {epoch:>5}/{args.epochs} step {step:>6} loss {loss_sum / len(xt):.4f} "
                      f"train {a('train'):.3f} id {a('id'):.3f} val {a('val'):.3f} test {a('test'):.3f} "
                      f"valCE {res['val']['cross_entropy']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if epoch in ckpt_epochs:
            state = {"epoch": epoch, "step": step, "model_state_dict": model.state_dict(),
                     "config": MODEL_CONFIGS[model_name], "run": out.name}
            if args.save_optimizer:
                state["optimizer_state_dict"] = opt.state_dict()
            torch.save(state, out / f"checkpoint_epoch_{epoch:04d}.pt")

    res, correct = evaluate_sets(model, sets, device)
    result = {
        "config": cfg, "config_hash": cfg_hash, "split_report": data.split_report,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "baselines": baselines, "final": res,
        "breakdown_val": breakdown(data, data.val_pairs, correct["val"]),
        "breakdown_test": breakdown(data, data.test_pairs, correct["test"]),
        "selected_by_val": best_by_val(history),
        "first_epoch_ge_90": {n: first_epoch_ge(history, n, 0.9) for n in ("train", "id", "val", "test")},
        "first_epoch_ge_50": {n: first_epoch_ge(history, n, 0.5) for n in ("train", "id", "val", "test")},
        "elapsed_seconds": time.time() - t0, "device": str(device), "history": history,
    }
    (out / "metrics.json").write_text(json.dumps(jsonable(result), indent=2))
    torch.save({"model_state_dict": model.state_dict(), "config": MODEL_CONFIGS[model_name],
                "summary": jsonable({k: v for k, v in result.items() if k != "history"})}, out / "final_model.pt")
    bt = result["breakdown_test"]
    print(f"  FINAL train {res['train']['accuracy']:.4f} id {res.get('id', {}).get('accuracy', float('nan')):.4f} "
          f"val {res['val']['accuracy']:.4f} test {res['test']['accuracy']:.4f} | test per-function mean "
          f"{bt['unique_function_mean']:.4f} ({bt['functions_solved_ge_90pct']}/{bt['n_unique_functions']} solved, "
          f"{bt['functions_near_chance_le_10pct']} near chance) | test CE {res['test']['cross_entropy']:.3f} "
          f"ECE {res['test']['ece']:.3f}", flush=True)
    print(f"  baselines(test): rectangle full={baselines['test']['rectangle_solver_full_tables']['accuracy']:.3f} "
          f"matched={baselines['test']['rectangle_solver_matched_cells']['accuracy']:.3f} "
          f"shortcut max={max(baselines['test']['shortcut_accuracy'].values()):.3f} chance={1 / 32:.3f}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["smoke", "train", "sweep"], default="smoke")
    p.add_argument("--model", "--models", dest="models", nargs="+", choices=list(MODEL_CONFIGS), default=None)
    p.add_argument("--condition", "--conditions", dest="conditions", nargs="+", choices=["fixed", "changing"],
                   default=["fixed"])
    p.add_argument("--weight-decay", dest="weight_decays", type=float, nargs="+", default=None)
    p.add_argument("--task-seed", dest="task_seeds", type=int, nargs="+", default=[0])
    p.add_argument("--split-seed", dest="split_seeds", type=int, nargs="+", default=[0])
    p.add_argument("--train-seed", dest="train_seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=None, help="default: 3 (smoke) / 600 (train, sweep)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--id-holdout", type=float, default=0.15, help="fraction of states withheld per trained function")
    p.add_argument("--min-slot-coverage", type=int, default=8)
    p.add_argument("--q-early", type=float, default=0.9)
    p.add_argument("--q-late", type=float, default=0.1)
    p.add_argument("--shift-epoch", type=int, default=None, help="default: epochs // 2")
    p.add_argument("--ramp-epochs", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--outdir", default="runs")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=0, help="0 = dense log-spaced schedule")
    p.add_argument("--save-optimizer", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument("--steps-per-sec", type=float, default=9.0, help="for --estimate-only (v1 CPU run: ~9 steps/s)")
    args = p.parse_args()

    if args.epochs is None:
        args.epochs = 3 if args.mode == "smoke" else 600
    if args.mode == "smoke":
        args.eval_every, args.log_every = 1, 1
    models = args.models or (list(MODEL_CONFIGS) if args.mode == "sweep" else ["m1"])
    wds = args.weight_decays or ([0.0, 0.01] if args.mode == "sweep" else [0.0])
    grid = list(itertools.product(models, args.task_seeds, args.split_seeds, args.train_seeds,
                                  args.conditions, wds))
    if "changing" in args.conditions and abs(args.q_early - 0.5) < 1e-9 and abs(args.q_late - 0.5) < 1e-9:
        print("note: q_early = q_late = 0.5 is the unmodified full table, i.e. identical to --condition fixed")

    d0 = WorldData(args.task_seeds[0], args.split_seeds[0], args.test_fraction, args.val_fraction,
                   args.id_holdout, args.min_slot_coverage)
    n_epoch = sum(len(c) for c in d0.train_cells.values())
    steps = math.ceil(n_epoch / args.batch_size) * args.epochs
    print(f"{len(grid)} run(s); ~{n_epoch} examples/epoch -> {steps} optimizer steps/run; "
          f"estimate {len(grid) * steps / args.steps_per_sec / 3600:.1f} h total at {args.steps_per_sec:g} steps/s")
    print(f"split (task {args.task_seeds[0]}, split {args.split_seeds[0]}): {d0.split_report}")
    if args.estimate_only:
        return
    for model_name, ts, ss, tr, cond, wd in grid:
        train_one(args, model_name, ts, ss, tr, cond, wd)


if __name__ == "__main__":
    main()
