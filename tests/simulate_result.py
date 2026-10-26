"""
Torch-free test of result assembly, JSON serialisation and summarize.py: replaces the network with random logits
but runs the same evaluation, breakdown, selection and serialisation code that run_p0.py uses.
    python tests/simulate_result.py && python summarize.py --runs /tmp/p0_sim
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import breakdown, evaluate_all_baselines  # noqa: E402
from data import WorldData  # noqa: E402
from utils import best_by_val, d_region_accuracy, first_epoch_ge, jsonable, merge_meta, metrics_from_logits  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/p0_sim")
rng = np.random.default_rng(0)
for train_seed in (0, 1):
    d = WorldData(0, 0)
    sets = d.evaluation_sets()

    def fake_eval(skill):
        res, correct = {}, {}
        for n, s in sets.items():
            logits = rng.normal(size=(len(s["y"]), 32))
            hit = rng.random(len(s["y"])) < skill[n]
            logits[np.arange(len(s["y"])), s["y"]] += np.where(hit, 8, 0)
            res[n], correct[n] = metrics_from_logits(logits, s["y"])
        seen = ["train", "id"]
        res["d_regions"] = d_region_accuracy(np.concatenate([correct[n] for n in seen]),
                                             merge_meta(*[sets[n]["meta"] for n in seen]))
        return res, correct

    history = []
    for epoch in (1, 5, 10, 20):
        skill = {"train": min(1, epoch / 15), "id": min(1, epoch / 25), "val": min(.5, epoch / 40), "test": min(.4, epoch / 50)}
        res, _ = fake_eval(skill)
        history.append({"epoch": epoch, "step": epoch * 39, "lr": 1e-3, "eval": res,
                        "exposure": {"q_target": None, "d_second_exposure": .5}})
    res, correct = fake_eval({"train": 1, "id": .9, "val": .5, "test": .4})
    result = {"config": {"model": "m1", "condition": "fixed", "weight_decay": 0.0, "epochs": 20, "task_seed": 0,
                         "split_seed": 0, "train_seed": train_seed, "q_early": .9, "q_late": .1, "shift_epoch": 10},
              "baselines": evaluate_all_baselines(d, train_seed), "final": res,
              "breakdown_val": breakdown(d, d.val_pairs, correct["val"]),
              "breakdown_test": breakdown(d, d.test_pairs, correct["test"]),
              "selected_by_val": best_by_val(history),
              "first_epoch_ge_90": {n: first_epoch_ge(history, n, .9) for n in ("train", "id", "val", "test")},
              "first_epoch_ge_50": {n: first_epoch_ge(history, n, .5) for n in ("train", "id", "val", "test")},
              "history": history}
    run = out / f"sim_train{train_seed}"
    run.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(result), indent=2, allow_nan=False)  # strict JSON: no NaN/Infinity
    (run / "metrics.json").write_text(text)
    print(f"wrote {run / 'metrics.json'} ({len(text) // 1024} KB), strict-JSON OK; "
          f"test per-function {result['breakdown_test']['unique_function_mean']:.3f}")
