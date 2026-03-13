## 1) Variation operators (quantale-inspired)

### 1.1 `Variation_quantale/crossover.py`

Implements **mask-based crossover** on top-level features.

#### `VariationQuantale`
- Treats each parent’s top-level features as a set.
- Builds `universe = m1_features ∪ m2_features`
- Generates random mask `p`:
  - if gene has STV: `prob = clamp((strength+confidence)/2, 0.1..0.9)`
  - else `prob = 0.5`
- Complement mask `p_comp = unit \ p`

Quantale-style operations:
- `join(A,B)` = union
- `product(A,mask)` = intersection
- `unit()` = universe

Crossover formula:
- child_features = (m1 ∩ p) ∪ (m2 ∩ p_comp)
- Preserves a stable ordering via `reference_order`
- Root operator follows parent1’s root (AND/OR)
- Produces new `Instance` with inferred knob list from tokens

`crossTopOne(instances, stv_dict, target_vals)`:
- Takes best instance and crosses it with others, producing children.

### 1.2 `Variation_quantale/mutation.py`

Implements two mutation styles:

#### Multiplicative mutation (pruning)
`product(expression_str)` recursively decides to keep/prune:
- For symbols: keep with prob derived from composite score or fallback mutation_rate
- For blocks: may prune whole block; otherwise recurse into children
- Rebuilds expression with original operator

`execute_multiplicative()`:
- Applies `product` to each top-level feature
- If none survive, returns `(base_op)`
- Returns a new `Instance`

#### Additive mutation (flipping/negating)
`join(feature)` toggles negation:
- `C` ↔ `(NOT C)`

`_mutate_expression(expr)` recursively flips symbols/blocks with probability scaled by `(1.1 - score)`

`execute_additive()`:
- Applies mutation across features and rebuilds expression

Both mutation modes optionally use ENF reduction (`reduce`) + `MeTTa()` to canonicalize after modification (as suggested by imports and surrounding integration).

---