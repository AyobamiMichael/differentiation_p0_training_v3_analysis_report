"""
v2 data module (numpy only; no torch import so it can be unit-tested anywhere).

Splits
    train / val / test pairs are separated by COMPLETE composite-function signature (world/splits.py).
    ID holdout: for every training composite function, a fixed set of states is withheld from training
    (chosen per signature, so functionally equal pairs such as (A1,A2)/(A2,A1) or (R1,k)/(R2,k) share it).
    Those (pair, state) cells form the "id" evaluation set: familiar function, unseen state.

Sampler (one sampler for every condition; budgets are matched)
    * Every training pair contributes exactly |allowed states| examples per epoch in every condition.
    * condition="fixed": the full table of allowed cells, every epoch (this is also the no-shift control).
    * condition="changing": pairs whose operand to D1/D2 is controlled are resampled so that the fraction of
      D-operand exposure falling in the shared region follows q(epoch): q_early before shift_epoch,
      q_late afterwards (optionally ramped). The operand is x when D is the FIRST operator and z=f_i(x)
      when D is the SECOND operator; pairs with D in both slots use a product of the two region weights.
    * q = 0.5 is defined as the unmodified full table, so changing(q_early=q_late=0.5) == fixed exactly.
      Constant q (e.g. 0.9/0.9 or 0.1/0.1) gives the high-/low-overlap worlds.
    * Pair counts are identical across conditions, so operator marginals are identical.
"""
import hashlib
import json

import numpy as np

from world.operators import N_STATES, build_world, all_compositions
from world.splits import choose_functional_split, validate_split3

QUERY_TOKEN, VOCAB_SIZE, STATE_TOKEN_START = 48, 49, 16
D_NAMES = ("D1", "D2")


class WorldData:
    def __init__(self, task_seed=0, split_seed=0, test_fraction=.20, val_fraction=.10,
                 id_holdout=.15, min_slot_coverage=8):
        self.task_seed, self.split_seed = int(task_seed), int(split_seed)
        self.ops, self.d_shared = build_world(self.task_seed)
        self.op_names = list(self.ops)
        self.op_to_id = {n: i for i, n in enumerate(self.op_names)}
        self.compositions = all_compositions(self.ops)
        self.train_pairs, self.val_pairs, self.test_pairs = choose_functional_split(
            self.compositions, self.op_names, test_fraction, val_fraction, self.split_seed, min_slot_coverage)
        self.split_report = validate_split3(self.compositions, self.train_pairs, self.val_pairs,
                                            self.test_pairs, self.op_names, min_slot_coverage)
        rng = np.random.default_rng(100_000 + self.task_seed)  # unchanged from v1: same token relabelling
        self.state_to_token = rng.permutation(N_STATES).astype(int)
        self.train_pairs_set, self.test_pairs_set = set(self.train_pairs), set(self.test_pairs)
        self._arr = {n: np.asarray(op.mapping, dtype=np.int64) for n, op in self.ops.items()}
        self._shared_mask = np.zeros(N_STATES, dtype=bool)
        self._shared_mask[list(self.d_shared)] = True
        self.id_holdout_fraction = float(id_holdout)
        self._build_cells()
        self._full_cache = None

    # ------------------------------------------------------------------ cells / ID holdout
    def _build_cells(self):
        n_hold = int(round(self.id_holdout_fraction * N_STATES))
        hrng = np.random.default_rng([300_000 + self.task_seed, self.split_seed])
        sigs = sorted({self.compositions[p] for p in self.train_pairs})
        self.holdout_by_signature = {
            sig: (np.sort(hrng.choice(N_STATES, size=n_hold, replace=False)) if n_hold else np.array([], dtype=np.int64))
            for sig in sigs}
        allx = np.arange(N_STATES)
        self.id_cells, self.train_cells = {}, {}
        for p in self.train_pairs:
            h = self.holdout_by_signature[self.compositions[p]]
            self.id_cells[p] = h
            self.train_cells[p] = np.setdiff1d(allx, h)

    # ------------------------------------------------------------------ example construction
    def _encode(self, pair, x):  # kept for backward compatibility
        i, j = pair
        return [self.op_to_id[i], self.op_to_id[j], STATE_TOKEN_START + int(self.state_to_token[int(x)]), QUERY_TOKEN]

    def _pair_block(self, pair, xs):
        i, j = pair
        xs = np.asarray(xs, dtype=np.int64)
        z = self._arr[i][xs]
        y = self._arr[j][z]
        tok = np.stack([np.full_like(xs, self.op_to_id[i]), np.full_like(xs, self.op_to_id[j]),
                        STATE_TOKEN_START + self.state_to_token[xs], np.full_like(xs, QUERY_TOKEN)], 1)
        return tok, self.state_to_token[y].astype(np.int64), xs, z

    def _examples(self, cells):
        toks, ys, meta = [], [], {k: [] for k in ("i_is_d", "j_is_d", "x_shared", "z_shared")}
        for pair, xs in cells.items():
            tok, y, xs, z = self._pair_block(pair, xs)
            toks.append(tok), ys.append(y)
            meta["i_is_d"].append(np.full(len(xs), pair[0] in D_NAMES))
            meta["j_is_d"].append(np.full(len(xs), pair[1] in D_NAMES))
            meta["x_shared"].append(self._shared_mask[xs])
            meta["z_shared"].append(self._shared_mask[z])
        cat = lambda a, dt: np.concatenate(a).astype(dt) if a else np.zeros((0,), dt)
        return (np.concatenate(toks).astype(np.int64), cat(ys, np.int64),
                {k: cat(v, bool) for k, v in meta.items()})

    def all_examples(self, pairs):
        """All 32 states for each pair, ordered by pair then state (alignment contract for breakdown())."""
        X, Y, _ = self._examples({p: np.arange(N_STATES) for p in pairs})
        return X, Y

    def evaluation_sets(self):
        full = lambda pairs: {p: np.arange(N_STATES) for p in pairs}
        spec = [("train", self.train_cells), ("val", full(self.val_pairs)), ("test", full(self.test_pairs))]
        if self.id_holdout_fraction > 0:
            spec.insert(1, ("id", self.id_cells))
        sets = {}
        for name, cells in spec:
            X, Y, meta = self._examples(cells)
            sets[name] = {"x": X, "y": Y, "meta": meta, "pairs": list(cells)}
        return sets

    def evaluation_data(self):  # v1-compatible: ((train_x, train_y), (test_x, test_y))
        s = self.evaluation_sets()
        return (s["train"]["x"], s["train"]["y"]), (s["test"]["x"], s["test"]["y"])

    # ------------------------------------------------------------------ training sampler
    @staticmethod
    def exposure_at(epoch, condition, q_early, q_late, shift_epoch, ramp_epochs=0):
        """Target shared-region exposure for D operands at a 0-based epoch (None for the fixed condition)."""
        if condition == "fixed":
            return None
        if condition != "changing":
            raise ValueError("condition must be 'fixed' or 'changing'")
        if epoch < shift_epoch:
            return float(q_early)
        if ramp_epochs <= 0:
            return float(q_late)
        a = min(1.0, (epoch - shift_epoch) / ramp_epochs)
        return float((1 - a) * q_early + a * q_late)

    def _region_weights(self, operands, q):
        sh = self._shared_mask[np.asarray(operands, dtype=np.int64)]
        n_sh, n_div = int(sh.sum()), int((~sh).sum())
        if n_sh == 0 or n_div == 0:
            return np.ones(len(sh))
        return np.where(sh, q / n_sh, (1.0 - q) / n_div)

    def training_examples(self, epoch=0, condition="fixed", q_early=.9, q_late=.1, shift_epoch=30,
                          ramp_epochs=0, seed=0):
        q = self.exposure_at(epoch, condition, q_early, q_late, shift_epoch, ramp_epochs)
        full_table = q is None or abs(q - .5) < 1e-9
        if full_table and self._full_cache is not None:
            return self._full_cache
        rng = np.random.default_rng([int(seed), int(epoch), 7])
        toks, ys = [], []
        st = {"d_first_shared": 0, "d_first_total": 0, "d_second_shared": 0, "d_second_total": 0, "unique_cells": 0}
        total_cells = 0
        for pair in self.train_pairs:
            i, j = pair
            allowed = self.train_cells[pair]
            di, dj = i in D_NAMES, j in D_NAMES
            if full_table or not (di or dj):
                chosen = allowed
            else:
                w = np.ones(len(allowed))
                if dj:
                    w = w * self._region_weights(self._arr[i][allowed], q)
                if di:
                    w = w * self._region_weights(allowed, q)
                chosen = rng.choice(allowed, size=len(allowed), replace=True, p=w / w.sum())
            tok, y, xs, z = self._pair_block(pair, chosen)
            toks.append(tok), ys.append(y)
            if di:
                st["d_first_total"] += len(xs)
                st["d_first_shared"] += int(self._shared_mask[xs].sum())
            if dj:
                st["d_second_total"] += len(xs)
                st["d_second_shared"] += int(self._shared_mask[z].sum())
            st["unique_cells"] += len(np.unique(xs))
            total_cells += len(allowed)
        X, Y = np.concatenate(toks), np.concatenate(ys)
        order = rng.permutation(len(X))
        info = {"q_target": q, "n_examples": int(len(X)),
                "d_first_exposure": st["d_first_shared"] / max(st["d_first_total"], 1),
                "d_second_exposure": st["d_second_shared"] / max(st["d_second_total"], 1),
                "unique_cell_coverage": st["unique_cells"] / max(total_cells, 1)}
        out = (X[order], Y[order], info)
        if full_table:
            self._full_cache = out
        return out

    # ------------------------------------------------------------------ provenance
    def fingerprint(self):
        payload = {"train": self.train_pairs, "val": self.val_pairs, "test": self.test_pairs,
                   "s2t": self.state_to_token.tolist(),
                   "ops": {n: list(o.mapping) for n, o in self.ops.items()},
                   "holdout": {str(k): v.tolist() for k, v in self.holdout_by_signature.items()}}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
