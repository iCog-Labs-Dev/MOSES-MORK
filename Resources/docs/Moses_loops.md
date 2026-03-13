## 1) MOSES loops (outer control)

### 1.1 `main.py` (unified entry point)

`run_moses(...)` chooses strategy based on `fg_type`:
- `"beta"` → `run_bp_moses(...)`
- `"alpha"` → `run_abp_moses(...)` then `_finalize_metapop(...)`
- otherwise defaults to alpha behavior

`main()`:
- seeds randomness
- loads truth table from CSV
- builds knobs from truth table (`knobs_from_truth_table`)
- creates exemplar `(AND)`
- evaluates exemplar fitness
- calls `run_moses(... fg_type="beta")`

### 1.2 `Moses/run_abp_moses.py` (alpha path)

Implements recursive outer loop:

Each iteration:
1. `sample_from_TTable(...)` → create demes near the current exemplar
2. For each deme: `run_deme_eda(...)` for `num_generations`
3. Merge best from each deme into `metapop` (deduplicate by value, keep best score)
4. Choose new exemplar:
   - default best in metapop
   - stagnation workaround: if unchanged, pick a random backup among top ranks
5. recurse with `max_iter - 1`

This is the “MOSES style” of:
- exploring neighborhoods (demes)
- learning distributions per deme (EDA)
- promoting best solutions into a global metapopulation

### 1.3 `Moses/run_bp_moses.py` (beta path)

Provides:
- `_finalize_metapop(...)` — dedupe, sort, print
- `run_variation(...)` — generation loop using:
  - `DependencyMiner.fit(values, weights)`
  - `BetaFactorGraph.add_dependency_rule(...)`
  - anchor prior from top rule
  - `run_evidence_propagation(...)`
  - create `stv_dict` from beta node beliefs
  - generate children with `crossTopOne(...)` if neighborhood big enough
  - generate mutation children (additive + multiplicative)
  - reduce + score new candidates (`reduce_and_score`)
  - extend deme instances with unique reduced candidates

`run_bp_moses(...)` wraps the above in an iterative/recursive MOSES control structure with termination conditions (max_iter, best_possible_score, etc.).

---
