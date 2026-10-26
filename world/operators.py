from dataclasses import dataclass
import numpy as np

N_STATES, N_OPERATORS = 32, 16

@dataclass(frozen=True)
class Operator:
    name: str
    mapping: tuple
    def __call__(self, x): return self.mapping[int(x)]

def _perm(values):
    values = tuple(int(v) for v in values)
    if len(values) != N_STATES or set(values) != set(range(N_STATES)):
        raise ValueError("Mapping is not a permutation of 0..31")
    return values

def cyclic(k): return _perm((x+k) % N_STATES for x in range(N_STATES))

def affine(a, b):
    if np.gcd(a, N_STATES) != 1: raise ValueError("a must be invertible mod 32")
    return _perm((a*x+b) % N_STATES for x in range(N_STATES))

def random_permutation(rng): return _perm(rng.permutation(N_STATES))

def partial_match_pair(rng, match_count=16):
    p = np.array(random_permutation(rng), dtype=int)
    states = np.arange(N_STATES); rng.shuffle(states)
    match_states, diff_states = states[:match_count], states[match_count:]
    q = p.copy()
    q[diff_states] = p[np.roll(diff_states, -1)]
    return tuple(p), _perm(q), frozenset(map(int, match_states))

def build_world(task_seed):
    rng = np.random.default_rng(task_seed); ops = {}
    for name, k in [("A1",1),("A2",3),("A3",7)]:
        ops[name] = Operator(name, cyclic(k))
    for name, a, b in [("B1",3,1),("B2",5,2),("B3",7,4)]:
        ops[name] = Operator(name, affine(a,b))
    for name in ("C1","C2"):
        ops[name] = Operator(name, random_permutation(rng))
    r = random_permutation(rng)
    ops["R1"] = Operator("R1", r); ops["R2"] = Operator("R2", r)
    g1,g2,_ = partial_match_pair(rng,16)
    ops["G1"] = Operator("G1",g1); ops["G2"] = Operator("G2",g2)
    while True:
        u1,u2 = random_permutation(rng),random_permutation(rng)
        if sum(a==b for a,b in zip(u1,u2)) <= 3: break
    ops["U1"] = Operator("U1",u1); ops["U2"] = Operator("U2",u2)
    d1,d2,d_shared = partial_match_pair(rng,16)
    ops["D1"] = Operator("D1",d1); ops["D2"] = Operator("D2",d2)
    return ops,d_shared

def compose(first, second): return _perm(second(first(x)) for x in range(N_STATES))
def all_compositions(ops):
    return {(i,j): compose(ops[i],ops[j]) for i in ops for j in ops}
