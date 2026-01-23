
from Representation.helpers import TreeNode, parse_sexpr, tokenize, isOP
from Representation.representation import Instance, Knob, Hyperparams

from typing import List
from copy import deepcopy
import random

def randomUniform(knobs):
    """
    Perform uniform random sampling to select knobs.
    """
    selected_knobs = []
    for knob in knobs:
        if random.random() > 0.5:
            selected_knobs.append(knob)

    return selected_knobs   

def randomBernoulli(p: float, instance: Instance, knob_perms: List, knobs: List[Knob]) -> Instance:
    """
    Perform Bernoulli sampling to select a knob for replacement.
    
    Args:
        p (float): Probability of selecting a knob.
        hyperparams: neihborhood_size identification.
        instance: The instance object (exemplar).
        knobs (List): List of knob objects to sample from.
        
    Returns:
        A set of newly generated instances.
    """
    new_instances = set()
    instanceExp = deepcopy(instance.value)
    sexp = tokenize(instanceExp)
    root = parse_sexpr(sexp)
    selected_knobs = randomUniform(knob_perms)
    if not selected_knobs:
        return

    candidates = []
    queue = [(root, [])]
    mutant_root = deepcopy(root)


    while queue:
        curr_node, curr_path = queue.pop(0)
        
        for i, child in enumerate(curr_node.children):
            child_path = curr_path + [i]
            candidates.append(child_path)
            
            if not child.is_leaf():
                queue.append((child, child_path))

    new_inst = Instance(
                value=str(mutant_root),
                id=instance.id + 1,
                score=0.0,
                knobs=deepcopy(instance.knobs)
            )
    
    for path in candidates:
        if random.random() < p:
            if not selected_knobs:
                break
            symbol = selected_knobs.pop(0)

            # Navigate to the PARENT of the target node using the path
            parent = mutant_root
            valid_path = True
            for idx in path[:-1]:
                if idx < len(parent.children):
                    parent = parent.children[idx]
                else:
                    valid_path = False
                    break
            
            target_idx = path[-1]
            if not valid_path or target_idx >= len(parent.children):
                continue  # Skip this mutation as the target no longer exists

            parent.children[target_idx] = TreeNode(symbol)

            mutant_value = str(mutant_root)
            
            if mutant_value == instanceExp:
                continue

            new_inst.value = mutant_value
            tokens = tokenize(symbol)
            for t in tokens:
                if isOP(t) or t in ['(', ')']:
                    continue

                knob = next((k for k in knobs if k.symbol == t), None)
                if knob and knob.symbol not in [k.symbol for k in new_inst.knobs]:
                    new_inst.knobs.append(knob)
            
    
    present_tokens = set(tokenize(new_inst.value))
    new_inst.knobs = [k for k in new_inst.knobs if k.symbol in present_tokens]

    return new_inst


def sample_new_instances(p: float, hyperparams: Hyperparams, instance: Instance, knob_perms: List, knobs: List[Knob]) -> List[Instance]:
    """
    Sample new instances using Bernoulli sampling.
    
    Args:
        p (float): Probability of selecting a knob.
        hyperparams: hyperparameters for sampling.
        instance: The instance object (exemplar).
        knobs (List): List of knob objects to sample from.
        
    Returns:
        A dict of newly generated instances.
    """
    new_instances = {}
    for _ in range(hyperparams.neighborhood_size):
        new_inst = randomBernoulli(p, instance, knob_perms, knobs)
        if new_inst and new_inst.value not in new_instances:
            new_instances[new_inst.value] = new_inst
    
    return new_instances