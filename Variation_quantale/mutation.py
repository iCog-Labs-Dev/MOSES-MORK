import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Representation.representation import (Quantale, Instance,
                                           Deme, Hyperparams,
                                           knobs_from_truth_table)
from Representation.csv_parser import load_truth_table
from Representation.helpers import tokenize, get_top_level_features, isSymbol
from reduct.enf.main import reduce
from hyperon import MeTTa
from typing import Any, Set
import random



class Mutation(Quantale):
    def __init__(self, instance: Instance, stv_values: dict, hyperparams: Hyperparams):
        super().__init__()
        self.instance = instance
        self.instance_value = instance.value
        self.base_op = self.get_base_OP()
        self.stv_values = stv_values
        self.mutation_rate = hyperparams.mutation_rate
        self.reference_order = get_top_level_features(self.instance_value)
        self.instance_vector = set(self.reference_order)


    def product(self, expression_str: str) -> str:
        """
        Recursively decides whether to keep or remove features.
        Returns: The pruned string, or None if removed completely.
        """
        if isSymbol(expression_str):
            score = self._get_composite_score(expression_str)
            prob_keep = score if score > 0 else self.mutation_rate
            
            if random.random() < prob_keep:
                return expression_str
            return None # Prune symbol

        score = self._get_composite_score(expression_str)
        
        prob_keep_block = score if score > 0 else 0.7
        if random.random() >= prob_keep_block:
             return None # Prune whole block
        
        children = get_top_level_features(expression_str)
        clean_str = expression_str.strip()
        op = "AND" if clean_str.startswith("(AND") else "OR" if clean_str.startswith("(OR") else self.base_op
        
        surviving_children = []
        for child in children:
            result = self.product(child)
            if result:
                surviving_children.append(result)
        
        if not surviving_children:
            return None
            
        return f"({op} {' '.join(surviving_children)})"

    def join(self, feature: str) -> str:
        """Transforms 'C' into '(NOT C)'."""
        if feature.startswith("(NOT ") and feature.endswith(")"):
            return feature[5:-1]
        return f"(NOT {feature})"
    
    def residium(self, parent1, parent2):
        return super().residium(parent1, parent2)
    
    def unit(self):
        return super().unit()
        
    def get_base_OP(self) -> str:
        content = self.instance_value.strip()
        if content.startswith("(") and content.endswith(")"):
            content = content[1:-1].strip()
            
        parts = content.split(' ', 1)
        return parts[0]
        
    def execute_multiplicative(self) -> str:
        """
        Recursive Multiplicative Mutation.
        Traverses tree and prunes weak branches/leaves.
        """
        keep_mask = []
        
        for feat in self.reference_order:
            result = self.product(feat)
            if result:
                keep_mask.append(result)
                
        if not keep_mask:
             return f"({self.base_op})"
             
        instance_exp = f"({self.base_op} {' '.join(keep_mask)})"
        new_instance = Instance(
            value=instance_exp,
            id=random.randint(1000, 9999),
            score=0.0,
            knobs=[k for k in self.instance.knobs if k.symbol in tokenize(instance_exp)]
        )
        return new_instance

    def _mutate_expression(self, expression_str: str) -> str:
        """
        Recursively mutates an expression string.
        1. If symbol: Check STV, maybe FLIP.
        2. If expression: 
           - Chance to Negate WHOLE expression.
           - Else: Recursively mutate inside.
        """
        if isSymbol(expression_str):
            score = self._get_composite_score(expression_str) 
            mutation_prob = self.mutation_rate * (1.1 - score)
            
            if random.random() < mutation_prob:
                return self.join(expression_str)
            return expression_str

        score = self._get_composite_score(expression_str)
        mutation_prob = self.mutation_rate * (1.1 - score)
        
        if random.random() < mutation_prob:
            return self.join(expression_str)
        
        children = get_top_level_features(expression_str)
        clean_str = expression_str.strip()
        if clean_str.startswith("(AND"):
            operator = "AND"
        elif clean_str.startswith("(OR"):
            operator = "OR"
        else:
            return expression_str
        
        mutated_children = [self._mutate_expression(child) for child in children]
        
        return f"({operator} {' '.join(mutated_children)})"

    def execute_additive(self, base_mutation_rate=None) -> str:
        """
        Additive Mutation: FLIPS genes or Mutates nested structures recursivley.
        """
        if base_mutation_rate:
            self.mutation_rate = base_mutation_rate

        new_features = []
        
        for feat in self.reference_order:
            mutated_feat = self._mutate_expression(feat)
            new_features.append(mutated_feat)

        instance_exp = f"({self.base_op} {' '.join(new_features)})"
        present_symbols = set(tokenize(instance_exp))
        parent_knobs = {k.symbol: k for k in self.instance.knobs}
        new_knobs = [parent_knobs[sym] for sym in present_symbols if sym in parent_knobs]
        
        new_instance = Instance(
            value=instance_exp,
            id=random.randint(1000, 9999),
            score=0.0,
            knobs=new_knobs
        )
        return new_instance
    
    
    def _get_composite_score(self, feature: str) -> float:
        """
        Calculates a score for a feature.
        If atomic (e.g., "A") and in STV, returns that score.
        If complex, averages scores of atomic children.
        """
        if feature in self.stv_values:
            s, c = self.stv_values[feature]
            return (s + c) / 2.0
            
        if feature.startswith("("):
            atoms = get_top_level_features(feature)
            scores = []
            for atom in atoms:
                if atom in self.stv_values:
                    s, c = self.stv_values[atom]
                    scores.append((s + c) / 2.0)
                else:
                    scores.append(self.mutation_rate)            
            if scores:
                return sum(scores) / len(scores)

        return self.mutation_rate






if __name__ == "__main__":
    hyperparam = Hyperparams(mutation_rate=0.3, crossover_rate=0.7, neighborhood_size=10, num_generations=10)
    knob_vals, target_vals = load_truth_table('example_data/test_bin.csv', output_col='O')
    knobs = knobs_from_truth_table(knob_vals)
    stv = {
    "A": (0.95, 0.95), # Strong
    "B": (0.80, 0.80), # Good
    "C": (0.20, 0.20),  # Weak
    "D": (0.35, 0.50),  # Weak
    "E": (0.46, 0.50),  # Weak
    "F": (0.46, 0.50),  # Weak
    "(OR D (AND E F))": (0.7, 0.5) # Moderate
    }
    # parent = "(AND A B C)"
    parent = Instance(value=f"(AND A B C (OR D (AND E F)))", id=0, score=0.0, knobs=knobs)

    metta = MeTTa()
    mutator = Mutation(parent, stv, hyperparam)
    print(f"Parent: {parent.value}")

    # 1. Generate Multiplicative Child (The Pruned One)
    # C is likely to be removed.
    child_mult = mutator.execute_multiplicative()
    print(f"Multiplicative Child: {child_mult.value} | Knobs: {[k.symbol for k in child_mult.knobs]}")

    # 2. Generate Additive Child (The Noisy One)
    # Might add (NOT A) or (NOT C)
    child_add = mutator.execute_additive()
    print(f"Additive Child:       {child_add.value} | Knobs: {[k.symbol for k in child_add.knobs]}")
    # reduced = reduce(metta, child_add.value)
    # print(f"Reduced Additive:     {reduced}")

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
