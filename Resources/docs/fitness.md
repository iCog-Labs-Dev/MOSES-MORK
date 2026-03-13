## Fitness, scoring, and population management

Fitness is referenced via `FitnessOracle` (imported from `Representation.representation` in `main.py` / MOSES loops). The details appear to be defined somewhere in `Representation/representation.py` beyond the snippet shown; the algorithm assumes:

- `fitness.get_fitness(instance)` produces `instance.score` in [0,1] (best_possible_score=1.0 used in beta mode termination).

Metapopulation (“metapop”):
- Top-level list collecting best individuals across iterations/demes.
- `_finalize_metapop` in `Moses/run_bp_moses.py` deduplicates by `inst.value`, sorts by score and complexity, prints top 10.

---