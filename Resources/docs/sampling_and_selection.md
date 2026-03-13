## 1) Sampling (deme creation)

### 1.1 `Representation/sampling.py`

This module creates demes around an exemplar and generates candidate expressions.

Key ideas:
- Generate logical “permutations” (new feature proposals) based on current operator:
  - If exemplar root op is AND, propose OR pairs; if OR, propose AND pairs.
- Uses feature selection (`Feature_selection_algo`) to pick which knobs/features to focus on.
- Uses ENF reduction via:
  - `from reduct.enf.main import reduce`
  - `MeTTa()` runtime (Hyperon)

Important functions:
- `sample_logical_perms(current_op, variables)` → candidate sub-expressions
- `randomUniform(...)` → select items probabilistically
- `randomBernoulli(...)` → mutate exemplar by probabilistically inserting/replacing features, prune duplicates, etc.

Also referenced:
- `sample_from_TTable(csv_path, hyperparams, exemplar, knobs, target, output_col='O')`
  - This is called by MOSES to produce demes.

> Conceptually, this is where the search neighborhood is built: you start with an exemplar and generate a set of “nearby” candidate programs grouped into demes.

---

## 2) Selection

### 2.1 `Representation/selection.py`

Two selection strategies:
- `select_top_k(deme, k)` — sort by `inst.score`, return first k
- `tournament_selection(deme, k, tournament_size)` — standard tournament selection

Used by:
- `Moses/run_bp_moses.py` for selecting exemplars for crossover/mutation.
- `run_abp_moses` uses `select_top_k` indirectly through EDA.

---