import math
from typing import List, Dict, Set, Tuple, FrozenSet, Union
from itertools import combinations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Representation.csv_parser import load_truth_table

def calculate_joint_entropy(features: List[List[bool]]) -> float:
    """
    Calculates joint entropy H(X1, X2, ..., Xn) for multiple boolean features.
    """
    if not features or not features[0]:
        return 0.0
    
    n_samples = len(features[0])
    
    # Count occurrences of each joint state
    state_counts: Dict[Tuple[bool, ...], int] = {}
    
    for i in range(n_samples):
        state = tuple(feature[i] for feature in features)
        state_counts[state] = state_counts.get(state, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    for count in state_counts.values():
        if count > 0:
            p = count / n_samples
            entropy -= p * math.log2(p)
    
    return entropy

def calculate_joint_mutual_information(feature_subset: List[List[bool]], target: List[bool]) -> float:
    """
    Calculates joint mutual information I(X1, X2, ..., Xn; Y).
    I(X1,...,Xn; Y) = H(Y) - H(Y | X1,...,Xn)
                    = H(X1,...,Xn) + H(Y) - H(X1,...,Xn, Y)
    """
    if not feature_subset:
        return 0.0
        
    h_features = calculate_joint_entropy(feature_subset)
    h_target = calculate_joint_entropy([target])
    h_joint = calculate_joint_entropy(feature_subset + [target])
    
    return h_features + h_target - h_joint

def calculate_conditional_mutual_information(
    new_features: List[List[bool]], 
    existing_features: List[List[bool]], 
    target: List[bool]
) -> float:
    """
    Calculates conditional mutual information I(New; Y | Existing).
    This measures how much new information the new features add given existing ones.
    
    I(New; Y | Existing) = H(Y | Existing) - H(Y | New, Existing)
                         = I(New, Existing; Y) - I(Existing; Y)
    """
    if not existing_features:
        return calculate_joint_mutual_information(new_features, target)
    
    mi_with_existing = calculate_joint_mutual_information(existing_features, target)
    mi_combined = calculate_joint_mutual_information(new_features + existing_features, target)
    
    return mi_combined - mi_with_existing

def calculate_interaction_gain(
    candidate_features: List[List[bool]],
    selected_features: List[List[bool]],
    target: List[bool]
) -> float:
    """
    Calculates the interaction gain: how much information the candidate adds
    beyond what's already captured by selected features.
    
    Returns: I(Candidate; Y | Selected) - Redundancy(Candidate, Selected)
    """
    # Relevance: conditional MI with target given selected features
    relevance = calculate_conditional_mutual_information(
        candidate_features, selected_features, target
    )
    # print(f"Calculated relevance (conditional MI): {relevance}")
    
    # Redundancy: how much the candidate overlaps with selected features
    if selected_features and candidate_features:
        # Average pairwise MI between candidate and each selected feature
        redundancy = 0.0
        for sel_feat in selected_features:
            # Calculate MI between candidate features and this selected feature
            mi = calculate_joint_mutual_information(
                candidate_features, 
                sel_feat
            ) 
            redundancy += abs(mi)
        redundancy /= len(selected_features)
    else:
        redundancy = 0.0
    # print(f"Relevance: {relevance}, Redundancy: {redundancy}, Interaction Gain: {relevance - redundancy}")
    
    return relevance - redundancy

def feature_order(csv_path: str, target_col: str) -> int:
    """
    A function that returns the practical order limit for feature selection
    Args:
        csv_path: Path to the CSV file.
        target_col: Name of the output/target column.
    Returns:
        practical_order: int
    """
    data_rows, _target_values = load_truth_table(csv_path, target_col)
    if not data_rows:
        # Keep a conservative default when the file is missing/empty/unreadable.
        num_features = 3
    else:
        # ``load_truth_table`` returns rows without the output column, so keys are features.
        num_features = len(data_rows[0].keys())

    return min(num_features, 4)
def interaction_aware_mrmr(
    csv_path: str, 
    target_col: str, 
    k: int = None,
    max_interaction_order: int = 2,
    output_type: str = 'list'
) -> Union[List[Tuple[FrozenSet[str], float]], Set[str], Set[Union[str, Tuple[str, ...]]]]:
    """

    Extended mRMR that considers higher-order feature interactions 

    Args:
        csv_path: Path to the CSV file.
        target_col: Name of the output/target column.
        k: Optional  Number of feature subsets to select if not specified or None - automatically stops when cumulative gain no longer increases.
        max_interaction_order: Maximum size of feature combinations to consider (1=single, 2=pairs, etc.)
        output_type: Format of the return value. 
            'list': Returns List[Tuple[FrozenSet[str], float]] (default).
            'set': Returns Set[str] (flattened set of all unique feature names).
            'subsets': Returns Set[Union[str, Tuple[str, ...]]] (set of selected subsets as strings or tuples).
        
    Returns:
        Depends on output_type.
    """

    # 1. Load Data
    data_rows, target_values = load_truth_table(csv_path, target_col)

    if not target_values or not data_rows:
        return []

    # Convert rows to column format
    columns: Dict[str, List[bool]] = {}
    for key in data_rows[0].keys():
        columns[key] = []

    for row in data_rows:
        for key, value in row.items():
            columns[key].append(value)

    valid_features = {
        name: vals for name, vals in columns.items()
        if len(vals) == len(target_values)
    }

    if not valid_features:
        return []

    # 2. Generate candidate subsets
    all_candidates: List[Tuple[FrozenSet[str], List[List[bool]]]] = []

    for order in range(1, min(max_interaction_order + 1, len(valid_features) + 1)):
        for feature_combo in combinations(valid_features.keys(), order):
            feature_set = frozenset(feature_combo)
            feature_data = [valid_features[name] for name in feature_combo]
            all_candidates.append((feature_set, feature_data))

    # 3. Compute initial relevance
    candidate_relevance: Dict[FrozenSet[str], float] = {}
    for feature_set, feature_data in all_candidates:
        mi = calculate_joint_mutual_information(feature_data, target_values)
        candidate_relevance[feature_set] = mi

    selected: List[Tuple[FrozenSet[str], float]] = []
    selected_feature_data: List[List[bool]] = []
    remaining_candidates = set(candidate_relevance.keys())

    cumulative_gain = 0.0

    # 4. First selection (highest relevance)
    if remaining_candidates:
        first = max(remaining_candidates, key=lambda fs: candidate_relevance[fs])
        first_score = candidate_relevance[first]

        if first_score > 0:
            selected.append((first, first_score))
            cumulative_gain += first_score
            selected_feature_data.extend([valid_features[name] for name in first])
            remaining_candidates.remove(first)

            # Remove subsets of selected
            remaining_candidates = {
                cand for cand in remaining_candidates
                if not cand.issubset(first)
            }

    # 5. Iterative selection with cumulative stopping
    while remaining_candidates:

        best_candidate = None
        best_score = float('-inf')

        for candidate_set in remaining_candidates:
            candidate_data = [valid_features[name] for name in candidate_set]

            score = calculate_interaction_gain(
                candidate_data,
                selected_feature_data,
                target_values
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate_set

        # Stop conditions
        if (
            best_candidate is None
            or cumulative_gain + best_score <= cumulative_gain
        ):
            # print(f" best_score: {best_score}, cumulative_gain: {cumulative_gain}")
            # print(f"Stopping at cumulative gain: {cumulative_gain:.4f}")
            break

        # Optional k-limit safeguard
        if k is not None and len(selected) >= k:
            # print(f"Reached k={k} selected subsets, stopping selection.")
            break

        # Select best candidate
        selected.append((best_candidate, best_score))
        cumulative_gain += best_score

        selected_feature_data.extend(
            [valid_features[name] for name in best_candidate]
        )

        remaining_candidates.remove(best_candidate)

        # Remove redundant subsets
        remaining_candidates = {
            cand for cand in remaining_candidates
            if not cand.issubset(best_candidate)
            and not any(cand.issubset(sel[0]) for sel in selected)
        }
        

    # 6. Output formatting
    if output_type == 'set':
        final_set = set()
        for fset, _ in selected:
            final_set.update(fset)
        return final_set

    elif output_type == 'subsets':
        final_subsets = set()
        for fset, _ in selected:
            if len(fset) == 1:
                final_subsets.add(list(fset)[0])
            else:
                final_subsets.add(tuple(sorted(fset)))
        return final_subsets
    
    return selected


# if __name__ == "__main__":
#     csv_file = "example_data/test_bin.csv"
#     target_column = "O"
    
#     print("=" * 60)
#     print("Interaction-Aware mRMR Feature Selection")
#     print("=" * 60)
#     print('')
    
#     # Test with different interaction orders
#     for max_order in [1, 2, 3]:
#         print(f"\nMax Interaction Order: {max_order}")
#         print("-" * 60)
        
#         results = interaction_aware_mrmr(
#             csv_file, 
#             target_column, 
#             k=0, 
#             max_interaction_order=max_order,
#             output_type='lists'
#         )
        
#         print(f"Selected Feature Set: {results}")
    
#     print("\n" + "=" * 60)