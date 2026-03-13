# Factor Graph - PLN

## 1) Factor-graph EDA (alpha path)

### 1.1 `FactorGraph_EDA/factor_graph.py`

Defines the factor graph used by the EDA loop:

- `SubtreeVariable(name, marginal_stv)`
  - `marginal_stv = (strength, confidence)` describing how likely a subtree is present
- `PairwiseFactor(var_a, var_b, stv, inferred=False)`
  - Stores in sorted order for stable keys
  - `stv` also `(strength, confidence)`
- `FactorGraph`
  - `variables: Dict[name, SubtreeVariable]`
  - `factors: Dict[(a,b), PairwiseFactor]`
  - `neighbors(var_name)` returns all adjacent factors
  - `neighbor_names(var_name)` returns adjacent variable names

This representation is used by `eda.py` to revise/deduce and sample.

### 1.2 `FactorGraph/pln.py`

Implements PLN truth-value algebra on STVs:
- `revision(stv1, stv2)` — merge evidence (confidence grows)
- `deduction(stv_ab, stv_bc, s_b)` — infer A→C from A→B and B→C
- `negation(stv)` — flip strength
- `modus_ponens(stv_a, stv_ab)` — infer B

Used in `FactorGraph_EDA/eda.py` to:
- revise variables/factors across generations
- deduce missing edges to “fill gaps” in factor graph structure

### 1.3 `FactorGraph_EDA/eda.py`

This file is the core of the alpha EDA pipeline.

It provides a staged workflow:

#### Stage 1: Miner → Factor graph
`build_factor_graph_from_miner(miner, dependencies)`:
- Variables come from `miner.single_weights`
  - strength = weight / total_weighted_contexts
  - confidence = `w2c(count)` (more observations = higher confidence)
- Factors come from “meaningful dependencies”
  - each becomes `PairwiseFactor(var_a, var_b, stv)`

#### Stage 2: Revision across generations
`revise_factor_graph(new_fg, old_fg)`:
- For shared variables/factors: apply `revision`
- For old-only items: carry forward with decayed confidence (×0.9)

#### Stage 3: Deduction
`apply_deduction(fg)`:
- For each pair of factors sharing a pivot variable (A‑B and B‑C), infer A‑C if missing.
- Mark inferred factors as `inferred=True` with lower confidence.

#### Stage 4+: Deme EDA loop (run_deme_eda)
`run_deme_eda(...)` is called by `run_abp_moses`:
- Select top k
- Mine dependencies (DependencyMiner)
- Build + revise + deduce factor graph
- Sample new candidates consistent with learned distribution
- Score and update deme
- Return best instance and factor graph

---

## 2) Beta factor graph (beta path)

### 2.1 BetaGraph / BetaFactorGraph (`FactorGraph_EDA/beta_bp.py`)

#### What it is (conceptually)
`BetaFactorGraph` is a lightweight belief network where each “node” is a feature/subtree name and its belief is represented as a **Beta distribution**.

Beta distributions are natural for modeling probabilities of Bernoulli-like events such as:

> “Is feature X present in good programs?”  
> “Should I include subtree Y during variation?”

Each node maintains a **BetaState** with:
- `alpha` (pseudo-count of positive evidence)
- `beta`  (pseudo-count of negative evidence)

From that it derives:
- `strength = alpha / (alpha + beta)` (mean probability)
- `confidence = (alpha + beta) / (alpha + beta + 1)` (saturating evidence amount)

### 2.2 What a “rule/factor” is in this beta graph
Instead of classic factor potentials, `BetaFactorGraph` stores a list of dependency rules:

```python
{ "src": ..., "dst": ..., "s": strength, "c": confidence }
```

These rules come directly from mined dependencies (the miner output). The idea is:

- If `src` is believed (high strength/confidence),
- then it should increase belief in `dst` (or otherwise shape the belief update),
- proportional to rule strength/confidence.

### 2.3 Key methods and their implementation roles

#### `add_dependency_rule(pair_str, rule_strength, rule_confidence)`
- Takes a string like `"A -- B"`
- Splits it into `(src, dst)`
- Adds/updates a rule record in `self.factors`

This is the main ingestion path from mined dependencies into the graph.

#### `set_prior(name, stv_strength, stv_confidence, base_counts=10.0)`
This is very important in `run_bp_moses`.

It “anchors” a node belief by converting an STV-style `(strength, confidence)` into Beta counts.

Typical strategy in this code:
- total evidence mass is something like `base_counts * f(confidence)`
- alpha ≈ mass * strength
- beta  ≈ mass * (1-strength)

So:
- High confidence gives larger total alpha+beta (harder to change)
- Strength biases toward inclusion vs exclusion

`run_bp_moses` uses `set_prior` to prevent the graph from floating at “uninformative” beliefs.

#### `run_evidence_propagation(...)`
This is the belief update loop:
- iteratively revises node Beta states based on incoming rules + neighbors
- may use damping and max iterations to stabilize

It produces updated node beliefs, which are then converted back into STV-like values (`stv_dict`) to guide crossover/mutation probabilities.

#### `visualize(...)`
Debugging/inspection aid (not essential to algorithm correctness).

### 2.4 What BetaFactorGraph achieves

1. **A probabilistic summary of which features are likely useful**
   Instead of hard “include/exclude,” it produces soft beliefs.

2. **Evidence accumulation**
   Beta counts accumulate evidence across propagation steps / generations.

3. **A mechanism to spread influence**
   Dependencies allow belief in one feature to increase/decrease belief in another (via rules).

4. **A stable numeric interface for variation**
   Eventually it produces an `stv_dict` so variation operators can do probability-based decisions per feature.

---
