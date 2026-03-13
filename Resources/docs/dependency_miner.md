
## Dependency mining

### `DependencyMiner/miner.py`

This discovers correlated program parts (subtrees / sibling co-occurrences).

There are two miners:

#### `OrderedTreeMiner`
- Enumerates **induced subtrees** rooted at each node (preserving parent-child)
- Counts document frequency (support) per expression tree
- `get_frequent_patterns()` returns patterns filtered by `min_support`

#### `DependencyMiner`
Main miner used in beta MOSES and EDA.
It focuses on **sibling co-occurrences inside a parent context**:
- For each non-leaf node with multiple children, treat children as a “context”
- Count:
  - single occurrences (`single_counts`, `single_weights`)
  - pair occurrences (`pair_counts`, `pair_weights`)
- Tracks `total_weighted_contexts` for normalization

The output of `DependencyMiner.get_meaningful_dependencies()` (called in `Moses/run_bp_moses.py`) is used to create rules/factors such as:
- `"A -- (NOT B)"` with `(strength, confidence)`

Those become edges in a factor graph / evidence model.

---