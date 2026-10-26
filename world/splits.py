from collections import defaultdict
import numpy as np

def choose_functional_ood_split(compositions, operator_names, test_fraction=.20, seed=0):
    groups = defaultdict(list)
    for pair, signature in compositions.items(): groups[signature].append(pair)
    rng = np.random.default_rng(seed); signatures = list(groups); rng.shuffle(signatures)
    target = max(1, round(len(signatures)*test_fraction))
    train, test = set(compositions), set()
    for sig in signatures:
        if len({compositions[p] for p in test}) >= target: break
        candidate = test | set(groups[sig]); remaining = train-candidate
        if all(any(op in pair for pair in remaining) for op in operator_names): test = candidate
    train -= test
    if {compositions[p] for p in train} & {compositions[p] for p in test}:
        raise AssertionError("Composite-function leakage detected")
    for op in operator_names:
        if not any(op in pair for pair in train): raise AssertionError(f"{op} absent from train")
    return sorted(train), sorted(test)

def validate_split(compositions, train, test, operator_names):
    train,test = set(train),set(test)
    assert not train & test and train | test == set(compositions)
    a={compositions[p] for p in train}; b={compositions[p] for p in test}
    assert not a & b
    assert all(any(op in pair for pair in train) for op in operator_names)
    return {"train_pairs":len(train),"test_pairs":len(test),
            "train_unique_functions":len(a),"test_unique_functions":len(b)}


# ----------------------------------------------------------------------------------------------
# v2: train / validation / test split by complete composite-function signature, with a per-slot
# coverage guarantee (every operator appears as FIRST and as SECOND operator in >= k training pairs).
# The test selection loop is identical to choose_functional_ood_split except for the stronger
# coverage rule, so for min_slot_coverage <= the achieved minimum it reproduces the same test set.
# ----------------------------------------------------------------------------------------------
def _slot_counts(pairs, operator_names):
    first = {o: 0 for o in operator_names}
    second = {o: 0 for o in operator_names}
    for i, j in pairs:
        first[i] += 1
        second[j] += 1
    return first, second


def _coverage_ok(pairs, operator_names, k):
    first, second = _slot_counts(pairs, operator_names)
    return all(first[o] >= k and second[o] >= k for o in operator_names)


def choose_functional_split(compositions, operator_names, test_fraction=.20, val_fraction=.10,
                            seed=0, min_slot_coverage=8):
    groups = defaultdict(list)
    for pair, signature in compositions.items():
        groups[signature].append(pair)
    rng = np.random.default_rng(seed)
    signatures = list(groups)
    rng.shuffle(signatures)
    n_sig = len(signatures)
    n_test = max(1, round(n_sig * test_fraction))
    n_val = round(n_sig * val_fraction)
    all_pairs = set(compositions)

    test, test_sigs = set(), []
    for sig in signatures:
        if len(test_sigs) >= n_test:
            break
        cand = test | set(groups[sig])
        if _coverage_ok(all_pairs - cand, operator_names, min_slot_coverage):
            test = cand
            test_sigs.append(sig)

    val, val_sigs, test_set = set(), [], set(test_sigs)
    for sig in signatures:
        if len(val_sigs) >= n_val:
            break
        if sig in test_set:
            continue
        cand = val | set(groups[sig])
        if _coverage_ok(all_pairs - test - cand, operator_names, min_slot_coverage):
            val = cand
            val_sigs.append(sig)

    train = all_pairs - test - val
    return sorted(train), sorted(val), sorted(test)


def validate_split3(compositions, train, val, test, operator_names, min_slot_coverage=8):
    train, val, test = set(train), set(val), set(test)
    assert not (train & val) and not (train & test) and not (val & test), "pair overlap between splits"
    assert train | val | test == set(compositions), "split does not cover all pairs"
    sig = lambda ps: {compositions[p] for p in ps}
    a, b, c = sig(train), sig(val), sig(test)
    assert not (a & b), "composite-function leakage: train/val"
    assert not (a & c), "composite-function leakage: train/test"
    assert not (b & c), "composite-function leakage: val/test"
    first, second = _slot_counts(train, operator_names)
    assert min(first.values()) >= min_slot_coverage, f"operator under-covered as FIRST: {first}"
    assert min(second.values()) >= min_slot_coverage, f"operator under-covered as SECOND: {second}"
    return {"train_pairs": len(train), "val_pairs": len(val), "test_pairs": len(test),
            "train_unique_functions": len(a), "val_unique_functions": len(b), "test_unique_functions": len(c),
            "min_train_pairs_as_first": min(first.values()), "min_train_pairs_as_second": min(second.values()),
            "min_slot_coverage_required": min_slot_coverage}
