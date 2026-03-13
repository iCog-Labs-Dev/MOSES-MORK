## 1) Canonicalization / ENF reduction (`reduct/enf`)

This subsystem provides a boolean-expression reducer inspired by “Elegant Normal Forms”.

### 1.1 `reduct/enf/main.py`

Defines a `reduce(metta, expr)` function and registers it as a Hyperon operation atom named `"reduce"`.

High-level reduce pipeline:
1. Parse incoming expression string (MeTTa format-like)
2. Convert to internal boolean tree via `BuildTree`
3. Wrap with a ROOT node
4. `propagateTruthValue(...)` converts expression into constraint tree by pushing a desired truth value downward
5. `gatherJunctors(...)` normalizes AND/OR structure into guard sets + children list representation
6. `reduceToElegance(...)` applies reduction rules (cuts, disconnects, detects tautology/contradiction, etc.)
7. Convert final constraint tree back to MeTTa expression string (`constraint_tree_to_metta_expr`)
8. Return parsed atoms back into MeTTa

### 1.2 Data structures & utilities
- `reduct/enf/DataStructures/Trees.py`:
  - NodeType enum
  - BinaryExpressionTreeNode
  - TreeNode (constraint tree with guardSet + children)
- `Utilities/BuildTree.py`: parses operator-form strings (`&`, `|`, `!`) into binary tree
- `Utilities/PropagateTruthValue.py`: pushes truth value down, flips operators under negation logic
- `Utilities/GatherJunctors.py`: merges/simplifies junctor nodes into normalized representation
- `Utilities/ReduceToElegance.py`: implements the actual reduction algorithm and signals:
  - `ReductionSignal`: DELETE / DISCONNECT / KEEP
  - applies cuts and consistency operations using helper set functions
- `Utilities/HelperFunctions.py`: printing, set operations, consistency checking, etc.

This reducer is used by sampling/mutation pipelines to keep expressions canonical and reduce redundancy.

---