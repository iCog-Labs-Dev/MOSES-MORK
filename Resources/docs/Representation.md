## `Representation/representation.py`

This file defines the core types used everywhere else.

#### `Quantale` (abstract base class)
A conceptual interface for algebraic composition:
- `join(parent1, parent2)`
- `product(parent1, parent2)`
- `residium(parent1, parent2)`
- `unit()`

In practice:
- `Variation_quantale/*` uses these ideas concretely (sets + masks).
- `Deme` inherits `Quantale` but leaves most operations TODO.

#### `KnobVariable`
Represents a “hole” / decision point and its allowed domain. (More conceptual in current code.)

#### `Factor` + `FactorGraph`
A basic factor graph abstraction:
- `Factor.evaluate(...)` returns a potential value for assigned variable values.
- `FactorGraph.neighbors(var)` lists factors containing the variable.

> Note: the **main EDA factor-graph used by the algorithm** is implemented in `FactorGraph_EDA/factor_graph.py`, not this older/general `FactorGraph` class.

#### `Knob` (dataclass)
Represents an input symbol (like `A`, `B`, etc.) and its truth values.

Fields:
- `symbol: str`
- `id: int`
- `Value: List[bool]` (note capital V in code)

#### `Instance` (dataclass)
Represents a candidate program:
- `value`: S-expression string (e.g. `(AND A (NOT B))`)
- `id`: instance id
- `score`: fitness score
- `knobs`: list of `Knob` objects used

It also includes `_get_complexity()` which approximates program complexity by token-counting (used for tie-breaking).

#### `Hyperparams` (dataclass)
Controls mutation/crossover and sampling:
- mutation_rate
- crossover_rate
- num_generations
- neighborhood_size
- bernoulli_prob / uniform_prob (used in sampling)

#### `Deme`
A neighborhood of `Instance`s, tracked across generations, with optional `factor_graph`.

---
