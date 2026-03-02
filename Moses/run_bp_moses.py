from DependencyMiner.miner import DependencyMiner

from Representation.representation import *
from Representation.helpers import *
from Representation.csv_parser import load_truth_table
from Representation.selection import select_top_k, tournament_selection
from Representation.sampling import sample_from_TTable

from Variation_quantale.crossover import VariationQuantale, crossTopOne
from Variation_quantale.mutation import Mutation

from FactorGraph_EDA.beta_bp import BetaFactorGraph

import random
import math
from typing import List


def _finalize_metapop(metapop: List[Instance], fg_type=None) -> List[Instance]:
    """Helper to sort and print the final metapopulation."""
    print(f"\n--- Final Metapopulation ---")
    unique_meta = list({inst.value: inst for inst in metapop}.values())
    # Safely get complexity if the method exists, else default to 0
    def get_cpx(inst):
        return inst._get_complexity() if hasattr(inst, '_get_complexity') else 0
        
    sorted_meta = sorted(unique_meta, key=lambda x: (-x.score, get_cpx(x)))
    
    for inst in sorted_meta[:10]:
        print(f"Instance: {inst.value} | Score: {inst.score:.5f}")
    return sorted_meta


def clean_structure(expression: str) -> str:
    """
    Parses an instance value, removes duplicate children from AND/OR nodes,
    and returns the cleaned string.
    """
    try:
        def _prune(node):
            if not node.children:
                return
            for child in node.children:
                _prune(child)
            
            if node.label in ["AND", "OR"]:
                unique = []
                seen = set()
                for child in node.children:
                    s = str(child)
                    if s not in seen:
                        seen.add(s)
                        unique.append(child)
                node.children = unique

        root = parse_sexpr(expression)
        _prune(root)
        return str(root)
    except Exception as e:
        return expression


def run_variation(deme, fitness, hyperparams, target, min_xover_neighbors=5):
    bg = BetaFactorGraph()
    
    for generation in range(hyperparams.num_generations):
        print("-" * 60)
        print(f"\n--- Generation {generation + 1} ---")

        selected_exemplars = select_top_k(deme, k=7)
        values = [inst.value for inst in deme.instances]
        weights = [inst.score for inst in deme.instances]
        
        miner = DependencyMiner()
        miner.fit(values, weights)
        correlation = miner.get_meaningful_dependencies()
        
        print("-" * 50)    
        for row in correlation:
            bg.add_dependency_rule(row['pair'], row['strength'], row['confidence'])

        if len(correlation) > 0:
            top_rule = correlation[0]
            parts = top_rule['pair'].split(' -- ')
            if len(parts) > 0:
                root_node = parts[0]
                print(f"\nDynamically setting prior for '{root_node}' based on rule: {top_rule['pair']}")
                bg.set_prior(
                    root_node, 
                    stv_strength=top_rule['strength'], 
                    stv_confidence=top_rule['confidence']
                )
        else:
            print("No correlations found...")
            continue
        
        bg.run_evidence_propagation(steps=20)
        
        stv_dict = {name: (node.strength, node.confidence) for name, node in bg.nodes.items()}
        
        # Track existing values to prevent any duplicates
        existing_values = {inst.value for inst in deme.instances}
        new_candidates = []

        if len(deme.instances) >= min_xover_neighbors:
            raw_children = crossTopOne(selected_exemplars, stv_dict, target)
            for inst in raw_children:
                inst.value = clean_structure(inst.value)
                if inst.value not in existing_values:
                    new_candidates.append(inst)
                    existing_values.add(inst.value)
        else:
            print(f"Skipping crossover (neighborhood size {len(deme.instances)} < {min_xover_neighbors})")

        mut_parent = max(selected_exemplars, key=lambda x: x.score)
        mutation = Mutation(mut_parent, stv_dict, hyperparams)

        child1 = mutation.execute_additive()
        if isinstance(child1, Instance):
            child1.value = clean_structure(child1.value)
            if child1.value not in existing_values:
                child1.score = fitness.get_fitness(child1)
                new_candidates.append(child1)
                existing_values.add(child1.value)

        child2 = mutation.execute_multiplicative()
        if isinstance(child2, Instance):
            child2.value = clean_structure(child2.value)
            if child2.value not in existing_values:
                child2.score = fitness.get_fitness(child2)
                new_candidates.append(child2)
                existing_values.add(child2.value)

        deme.instances.extend(new_candidates)
    
    return deme

def run_bp_moses(exemplar: Instance, fitness: FitnessOracle, hyperparams: Hyperparams,
              target: List[bool], csv_path: str, metapop: List[Instance], 
              iteration: int = 1, max_iter: int = 30, 
              distance: int = 1, max_dist: int = 5, 
              last_chance: bool = False, best_possible_score: float = 1.0) -> List[Instance]:
    
    if max_iter <= iteration:
        print("\nMax iterations limit reached...")
        return _finalize_metapop(metapop)
    
    # if distance > max_dist:
    #     print("\nTerminating because maximum search distance reached...")
    #     return _finalize_metapop(metapop)

    if exemplar.score >= best_possible_score:
        print(f"\nTerminating because best possible score ({best_possible_score}) was found!")
        return _finalize_metapop(metapop)

    demes = sample_from_TTable(csv_path, hyperparams, exemplar, exemplar.knobs, target, output_col='O')
    print(f"\n[Iter {iteration} | Dist {distance}] Running Variation for {len(demes)} demes centered on: {exemplar.value}...")
    
    new_demes = [run_variation(deme, fitness, hyperparams, target) for deme in demes]

    print("Iteration: ", iteration)
    print("\n--- Top Instances from Each Deme ---")
    meta_dict = {inst.value: inst for inst in metapop}
    added_count = 0

    for i, deme in enumerate(new_demes):
        if not deme.instances: continue
        best_in_deme = max(deme.instances, key=lambda x: x.score)
        print(f"Deme {i} Best: {best_in_deme.value:<30} | Score: {best_in_deme.score:.4f}")
        
        if best_in_deme.value not in meta_dict or best_in_deme.score > meta_dict[best_in_deme.value].score:
            meta_dict[best_in_deme.value] = best_in_deme
            added_count += 1

    metapop = sorted(meta_dict.values(), key=lambda x: x.score, reverse=True)
    print(f"Merged {added_count} new unique instances. Metapop Size: {len(metapop)}")

    new_best = metapop[0]
    has_improved = new_best.score > exemplar.score + 1e-6 # epsilon for float comparison

    
    if has_improved:
        print(f"*** Improvement found! Score: {new_best.score:.4f} ***")
        next_exemplar = new_best
        next_distance = 1
        next_last_chance = False
    else:
        if not last_chance:
            print(f"(!) Stagnation at score {exemplar.score:.4f}. Trying one last chance at current center...")
            next_exemplar = exemplar
            next_distance = distance
            next_last_chance = True
        else:
            print(f"(!) No improvement after last chance. Expanding search distance...")
            next_distance = distance + 1
            next_last_chance = False
            
            # Jump to a backup exemplar if we are stuck locally
            if len(metapop) > 1:
                backup_index = random.randint(1, min(4, len(metapop)-1))
                next_exemplar = metapop[backup_index]
                print(f" -> Switching focus to rank {backup_index}: {next_exemplar.value[:20]}... (Score: {next_exemplar.score:.4f})")
            else:
                next_exemplar = exemplar

    # Recurse with updated state
    return run_bp_moses(
        next_exemplar, fitness, hyperparams, target, csv_path, metapop, 
        iteration=iteration + 1, max_iter=max_iter, 
        distance=next_distance, max_dist=max_dist, 
        last_chance=next_last_chance, best_possible_score=best_possible_score
    )

def run_bp_moses_sa(exemplar: Instance, fitness: FitnessOracle, hyperparams: Hyperparams,
                 target: List[bool], csv_path: str, metapop: List[Instance], 
                 iteration: int = 1, max_iter: int = 30, 
                 temperature: float = 1.0, cooling_rate: float = 0.9, 
                 best_possible_score: float = 1.0) -> List[Instance]:
    
    if iteration > max_iter:
        print("\nMax iterations limit reached...")
        return _finalize_metapop(metapop)
    
    if temperature < 1e-5:
        print("\nSystem has cooled down completely (T ≈ 0). Terminating...")
        return _finalize_metapop(metapop)

    if exemplar.score >= best_possible_score:
        print(f"\nTerminating because best possible score ({best_possible_score}) was found!")
        return _finalize_metapop(metapop)

    demes = sample_from_TTable(csv_path, hyperparams, exemplar, exemplar.knobs, target, output_col='O')
    print(f"\n[Iter {iteration} | Temp {temperature:.4f}] Running Variation for {len(demes)} demes centered on: {exemplar.value}...")
    
    new_demes = [run_variation(deme, fitness, hyperparams, target) for deme in demes]

    print("\n--- Top Instances from Each Deme ---")
    meta_dict = {inst.value: inst for inst in metapop}
    added_count = 0

    for i, deme in enumerate(new_demes):
        if not deme.instances: continue
        best_in_deme = max(deme.instances, key=lambda x: x.score)
        print(f"Deme {i} Best: {best_in_deme.value:<30} | Score: {best_in_deme.score:.4f}")
        
        if best_in_deme.value not in meta_dict or best_in_deme.score > meta_dict[best_in_deme.value].score:
            meta_dict[best_in_deme.value] = best_in_deme
            added_count += 1

    metapop = sorted(meta_dict.values(), key=lambda x: x.score, reverse=True)
    print(f"Merged {added_count} new unique instances. Metapop Size: {len(metapop)}")

    round_bests = [max(deme.instances, key=lambda x: x.score) for deme in new_demes if deme.instances]
    if not round_bests:
        print("No valid instances generated this round. Cooling and staying in place...")
        next_exemplar = exemplar
    else:
        new_best = max(round_bests, key=lambda x: x.score)
        
        delta_score = new_best.score - exemplar.score
        
        if delta_score > 1e-6:
            print(f"*** Strict Improvement found! Score: {new_best.score:.4f} (+{delta_score:.4f}) ***")
            next_exemplar = new_best
        
        else:
            acceptance_prob = math.exp(delta_score / temperature)
            random_draw = random.random() + 1e-3
            
            if abs(delta_score) <= 1e-6:
                print(f"--- SA ACCEPTED sideways move: {new_best.score:.4f} (Score didn't change) ---")
                next_exemplar = new_best
            
            if random_draw < acceptance_prob:
                print(f"--- SA ACCEPTED worse solution: {new_best.score:.4f} (Prob: {acceptance_prob:.4f}, Draw: {random_draw:.4f}) ---")
                next_exemplar = new_best
            else:
                print(f"(!) SA REJECTED worse solution: {new_best.score:.4f} (Prob: {acceptance_prob:.4f}, Draw: {random_draw:.4f})")
                print(f" -> Staying centered on {exemplar.value} (Score: {exemplar.score:.4f})")
                next_exemplar = exemplar

    next_temperature = temperature * cooling_rate

    return run_bp_moses_sa(
        next_exemplar, fitness, hyperparams, target, csv_path, metapop, 
        iteration=iteration + 1, max_iter=max_iter, 
        temperature=next_temperature, cooling_rate=cooling_rate, 
        best_possible_score=best_possible_score
    )
