import random
from .representation import Deme, Instance
from typing import List

def select_top_k(deme: Deme, k: int) -> List[Instance]:
    """
    select_top_k: Selects the top k instances from the deme based on their scores.
    Args:
        deme (Deme): The deme containing instances.
        k (int): The number of top instances to select.
    Returns: A list of the top k instances.
    """
    sorted_instances = sorted(deme.instances, key=lambda inst: inst.score, reverse=True)
    return sorted_instances[:k]

def tournament_selection(deme: Deme, k:int, tournament_size: int) -> List[Instance]:
    """
    tournament_selection: Selects instances from the deme using tournament selection.
    Args:
        deme (Deme): The deme containing instances.
        k (int): The number of instances to select.
        tournament_size (int): The size of each tournament.
    Returns: A list of selected instances.
    """
    selected_instances = []
    num_instances = len(deme.instances)

    K = min(k, num_instances)
    for _ in range(K):
        tournament = random.sample(deme.instances, min(tournament_size, num_instances))
        winner = max(tournament, key=lambda inst: inst.score)
        selected_instances.append(winner)

    return selected_instances