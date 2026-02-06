import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Representation.representation import (Quantale, Instance,
                                           Deme, Hyperparams,
                                           knobs_from_truth_table)
from Representation.csv_parser import load_truth_table
from Representation.sampling import sample_from_TTable
from Representation.helpers import tokenize, get_top_level_features
from typing import Any, Set
import random


class VariationQuantale(Quantale):
    def __init__(self, m1, m2, stv_values):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.stv_values = stv_values
        m1_list = get_top_level_features(m1.value)
        m2_list = get_top_level_features(m2.value)
        
        self.reference_order = list(dict.fromkeys(m1_list + m2_list))
        self.m1_vector = set(m1_list)
        self.m2_vector = set(m2_list)

        self.universe = self.m1_vector | self.m2_vector
        self.p = self._generate_random_mask(self.stv_values)
        self.p_comp = self.residium(self.p, self.unit())

    
    def _generate_random_mask(self, stv_values) -> Set[Any]:
        """Creates a random subset of the universe."""
        mask = set()
        for gene in self.universe:
            if gene in stv_values.keys():
                    strength, confidence = stv_values[gene]
                    prob = (strength + confidence) / 2.0
                    prob = max(0.1, min(0.9, prob))
            else:
                # Default for unknown genes: pure random (0.5)
                prob = 0.5

            if random.random() < prob:
                mask.add(gene)
                
        return mask
        
    
    def join(self, set_a: Set[Any], set_b: Set[Any]) -> Set[Any]:
        """
        The Sum Operation (+).
        Logic: Union (OR).
        Used to combine the two parent parts: Part1 + Part2
        """
        return set_a.union(set_b)
    
    def product(self, instance_features: Set[Any], mask: Set[Any]) -> Set[Any]:
        """
        The Product Operation (tensor).
        Logic: Intersection (AND).
        Used to apply mask: Parent * Mask
        """
        return instance_features.intersection(mask)

    def residium(self, a: Set[Any], unit: Set[Any]) -> Set[Any]:
        """
        The Implication Operation (->).
        Logic: Relative Complement / Implication.
        For boolean sets: a -> b is equivalent to (~a U b).
        
        Used here specifically to calculate the Complement Mask:
        p_comp = p -> 0 (or Unit - p depending on exact algebra definition)
        Simple set difference works for the 'Complement' concept here.
        """
        # Calculating p_comp as (Universe - p)
        # return self.universe.difference(a)
        return unit.difference(a)


    def unit(self) -> Set[Any]:
        """Returns 'e' (The Top Element / Identity). In this case, the whole Universe."""
        return self.universe
    
    def zero(self) -> Set[Any]:
        """Returns '0' (The Bottom Element). An empty set."""
        return set()
    
    def execute_crossover(self) -> Set[Any]:
        """
        Runs the formula: Child = (m1 * p) + (m2 * ~p)
        """
        # Parent 1 Part (m1 AND p)
        part1 = self.product(self.m1_vector, self.p)
        
        # Parent 2 Part (m2 AND p_comp)
        part2 = self.product(self.m2_vector, self.p_comp)
        
        # Join (part1 OR part2)
        child_unordered_set = self.join(part1, part2)
        ordered_child_features = [k for k in self.reference_order if k in child_unordered_set]
        root_op = "AND" 
        if self.m1.value.strip().startswith("(OR"):
            root_op = "OR"
            
        child_value = f"({root_op} {' '.join(ordered_child_features)})"
        pool_knobs = self.m1.knobs + self.m2.knobs
        unique_pool = {k.symbol: k for k in pool_knobs}
        
        child_knobs = []
        child_tokens = tokenize(child_value) 
        
        for token in set(child_tokens):
            if token in unique_pool:
                child_knobs.append(unique_pool[token])
                
        new_id = random.randint(1000, 9999)
        child_instance = Instance(
            value=child_value,
            id=new_id,
            score=0.0,
            knobs=child_knobs
        )
        
        return child_instance


# if __name__ == "__main__":
#     hyperparam = Hyperparams(mutation_rate=0.3, crossover_rate=0.7, neighborhood_size=10, num_generations=10)
#     knob_vals, target_vals = load_truth_table('example_data/test_bin.csv', output_col='O')
#     knobs = knobs_from_truth_table(knob_vals)

#     exemplar = Instance(value=f"(AND A B C D E F)", id=0, score=0.0, knobs=knobs)
    
    # demes = sample_from_TTable('example_data/test_bin.csv', hyperparam, exemplar, knobs, target_vals, output_col='O')
    # deme1 = demes[0]
    # inst1, inst2 = deme1.instances[0], deme1.instances[1]
    # print("--- Parent Instances ---")
    # print(f"Instance 1: {inst1.value} | Knobs: {[k.symbol for k in inst1.knobs]}")
    # print(f"Instance 1: {inst2.value} | Knobs: {[k.symbol for k in inst2.knobs]}")
    # inst1 = Instance(value=f"(AND A B C D (OR (NOT A) E) F)", id=0, score=0.0, knobs=knobs)
    # inst2 = Instance(value=f"(AND A B E F)", id=0, score=0.0, knobs=knobs)
    # stv1 = {"A": (0.7, 0.7), "B": (0.8, 0.8),"E": (0.2, 0.2), "F": (0.4, 0.4), "(OR (NOT A) E)": (0.6, 0.6)}

    # var_q = VariationQuantale(inst1, inst2, stv1)
    # print(f"Universe: {var_q.universe}")
    # print(f"reference order: {var_q.reference_order}")
    # print(f"p: {var_q.p}")
    # print(f"p_comp: {var_q.p_comp}")

    # test = (tokenize("(AND A B C (OR A B) D E F)")[2:-1])
    # test2 = " ".join(test)
    # print(test2)
    # child_instance = var_q.execute_crossover()
    # print("--- Child Instance Features after Crossover ---")
    # print(f"Child Instance: {child_instance.value} | Knobs: {[k.symbol for k in child_instance.knobs]}")