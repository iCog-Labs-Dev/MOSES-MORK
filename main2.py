import random

from Representation.representation import *
from Representation.helpers import *
from Representation.csv_parser import load_truth_table
from Representation.selection import select_top_k  
from Representation.eda import run_deme_eda
from Representation.sampling import sample_from_TTable


def run_moses(
    exemplar: Instance,
    fitness: FitnessOracle,
    hyperparams: Hyperparams,
    knobs: List[Knob],
    target: List[bool],
    csv_path: str,
    metapop: List[Instance],
    max_iter: int,
) -> List[Instance]:
    """
    Outer MOSES loop.

    Each iteration:
      1. Sample demes using feature selection algorithms.
      2. For each deme, run ``num_generations`` of EDA (mine → FG → PLN → sample).
      3. Collect the fittest instance from every deme into the metapopulation.
      4. Pick a new exemplar and recurse (or stop when *max_iter* is exhausted).
    """
    if max_iter <= 0:
        print("\nMax iterations reached.")
        print(f"\n{'='*60}")
        print("--- Final Metapopulation ---")
        unique_meta = {inst.value: inst for inst in metapop}.values()
        sorted_metapop = sorted(unique_meta, key=lambda x: -x.score)
        for inst in sorted_metapop[:10]:
            print(f"  {inst.value:<40} | Score: {inst.score:.5f}")
        return list(sorted_metapop)

    # --- 1. Sample demes centred on the current exemplar -------------------
    print(f"\n{'='*60}")
    print(f"[Iter {max_iter}] Exemplar: {exemplar.value}  (score={exemplar.score:.4f})")
    demes = sample_from_TTable(csv_path, hyperparams, exemplar, knobs, target, output_col='O')
    print(f"  Sampled {len(demes)} deme(s)")

    # --- 2. Run EDA generations on each deme -------------------------------
    best_instances: List[Instance] = []
    num_eda_gens = hyperparams.num_generations 

    for i, deme in enumerate(demes):
        if not deme.instances:
            print(f"  Deme {i}: empty, skipping")
            continue

        print(f"\n  --- Deme {i} ({len(deme.instances)} instances) ---")
        best_inst, _fg = run_deme_eda(
            deme,
            fitness,
            num_generations=num_eda_gens,
            top_k=min(5, len(deme.instances)),
            min_pmi=0.0,
            min_freq=1,
            sample_size=len(deme.instances),
            all_knobs=knobs,
            verbose=True,
        )
        if best_inst is not None:
            best_instances.append(best_inst)
            print(f"  Deme {i} best: {best_inst.value:<30} | Score: {best_inst.score:.4f}")

    # --- 3. Merge best instances into metapopulation -----------------------
    meta_dict = {inst.value: inst for inst in metapop}
    added_count = 0
    for inst in best_instances:
        if inst.value not in meta_dict or inst.score > meta_dict[inst.value].score:
            if inst.value not in meta_dict:
                added_count += 1
            meta_dict[inst.value] = inst

    metapop = sorted(meta_dict.values(), key=lambda x: x.score, reverse=True)
    print(f"\n  Merged {added_count} new unique instance(s). Metapop size: {len(metapop)}")

    # --- 4. Pick next exemplar ---------------------------------------------
    new_exemplar = metapop[0]

    # stagnation check: if best hasn't changed, try a different exemplar
    if new_exemplar.value == exemplar.value and len(metapop) > 1:
        backup_idx = random.randint(1, min(4, len(metapop) - 1))
        new_exemplar = metapop[backup_idx]
        print(f"  (!) Stagnation — switching to rank {backup_idx}: "
              f"{new_exemplar.value[:30]}  (score={new_exemplar.score:.4f})")

    return run_moses(
        new_exemplar, fitness, hyperparams, knobs, target,
        csv_path, metapop, max_iter - 1,
    )


def main():
    # random.seed(42)

    csv_path = "example_data/test_and_4vars.csv"
    hyperparams = Hyperparams(
        mutation_rate=0.3,
        crossover_rate=0.5,
        num_generations=20,
        neighborhood_size=15,
        bernoulli_prob=0.2,
        uniform_prob=0.2,
    )
    ### Loading the truth table and extract knobs
    input_rows, target = load_truth_table(csv_path, output_col='O')
    knobs = knobs_from_truth_table(input_rows)
    print(f"Knobs: {[k.symbol for k in knobs]}")

    # Start with an empty exemplar
    exemplar = Instance(value="(AND)", id=0, score=0.0, knobs=[])
    fitness = FitnessOracle(target)
    ### Score the initial exemplar
    exemplar.score = fitness.get_fitness(exemplar)
    print(f"Exemplar: {exemplar.value} | Score: {exemplar.score}")

    metapop = [exemplar]
    print('*' * 60)
    final_metapop = run_moses(
        exemplar, fitness, hyperparams, knobs, target,
        csv_path, metapop, max_iter=15,
    )
    print('*' * 60)
    print(f"\nTop results:")
    for inst in final_metapop[:10]:
        print(f"  {inst.value:<40} | Score: {inst.score:.5f}")


if __name__ == "__main__":
    main()