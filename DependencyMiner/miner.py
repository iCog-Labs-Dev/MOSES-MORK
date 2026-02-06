import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Representation.helpers import TreeNode, parse_sexpr, tokenize

import collections
import itertools
import math


def sigmoid(x, k=1.0):
    """Squashes any number into the [0, 1] interval."""
    return 1 / (1 + math.exp(-k * x))

class OrderedTreeMiner:
    def __init__(self, min_support=2):
        self.min_support = min_support
        self.patterns = collections.defaultdict(int)
        self.pattern_map = {} # Maps string repr -> TreeNode obj 

    def _get_subtrees(self, node):
        """
        Generates all 'Induced' subtrees rooted at this node.
        An induced subtree must preserve parent-child relationships.
        
        Returns a list of strings representing subtrees rooted here.
        """
        my_subtrees = [f"({node.label})"]
        
        # Get all subtrees for all children
        child_subtree_groups = [self._get_subtrees(child) for child in node.children]
        
        # Simple Logic: We will form a subtree by taking the current node
        # and attaching ONE of the valid subtrees from each child (or skipping the child).
        # STRICT ORDERED MINING: We preserve the order of children.
        
        
        # For every child, we can either:
        # a) Not include it (None)
        # b) Include one of its subtrees
        
        # Generate cartesian product of children's possibilities
        # [''] means we're just skipping the child
        options_per_child = [[''] + group for group in child_subtree_groups]
        
        for combination in itertools.product(*options_per_child):
            valid_children = [c for c in combination if c]
            
            if not valid_children:
                continue
                
            children_str = " ".join(valid_children)
            subtree_str = f"({node.label} {children_str})"
            my_subtrees.append(subtree_str)
            
        return my_subtrees

    def fit(self, s_expressions):
        """
        Process the list of s-expressions and find frequent patterns.
        """
        self.patterns = collections.defaultdict(int)
        
        for expr in s_expressions:
            tokens = tokenize(expr)
            root = parse_sexpr(tokens)
            
            # To count support correctly (document frequency), we use a set per tree
            seen_in_this_tree = set()
            
            # Generating subtrees for every node
            queue = [root]
            while queue:
                curr = queue.pop(0)
                
                # Generate all subtrees rooted at current node
                subtrees = self._get_subtrees(curr)
                seen_in_this_tree.update(subtrees)
                
                queue.extend(curr.children)
            
            # Update global counts
            for pattern in seen_in_this_tree:
                self.patterns[pattern] += 1
                
        return self

    def get_frequent_patterns(self):
        """Returns sorted list of (pattern, count) tuples."""
        # Filter by min_support
        # Sort by frequency (desc), then length (desc)
        frequent = {k: v for k, v in self.patterns.items() if v >= self.min_support}
        return sorted(frequent.items(), key=lambda item: (-item[1], -len(item[0])))


# The PMI score of should be weighted by the fitness score of the instances.

class DependencyMiner:
    def __init__(self):
        self.pair_counts = collections.defaultdict(int)
        self.single_counts = collections.defaultdict(int)
        self.pair_weights = collections.defaultdict(float)
        self.single_weights = collections.defaultdict(float)
        self.total_weighted_contexts = 0.0
        self.total_count = 0

    def _get_canonical(self, node):
        """Returns a string representation of a node (simplified for mining keys)."""
        return str(node)

    def fit(self, s_expressions, weights):
        """
        Scans the trees specifically looking for SIBLING CO-OCCURRENCES.
        This detects which knobs/arguments are coupled.
        """
        for expr, weight in zip(s_expressions, weights):
            if weight <= 0: continue

            tokens = tokenize(expr)
            root = parse_sexpr(tokens)
            
            # BFS/DFS traversal to find every "Context" (Parent node)
            queue = [root]
            while queue:
                current = queue.pop(0)
                
                # Only considering non-leaf nodes with multiple children as contexts
                if not current.is_leaf() and len(current.children) > 1:
                    self.total_weighted_contexts += weight
                    self.total_count += 1
                    
                    child_keys = [self._get_canonical(c) for c in current.children]
                    
                    # Counting individual occurrences, which aviods duplicates (Mariginal).
                    # We can also use the frequency
                    seen_in_context = set()
                    for k in child_keys:
                        if k not in seen_in_context:
                            self.single_weights[k] += weight
                            self.single_counts[k] += 1
                            seen_in_context.add(k)
                    
                    # Counting pairs
                    for i in range(len(child_keys)):
                        for j in range(i + 1, len(child_keys)):
                            k1, k2 = child_keys[i], child_keys[j]
                            if k1 > k2: k1, k2 = k2, k1 # Sort for consistency
                            
                            self.pair_weights[(k1, k2)] += weight
                            self.pair_counts[(k1, k2)] += 1
                
                queue.extend(current.children)
        return self

    def get_meaningful_dependencies(self, min_pmi=0.1, min_weight=0.01, min_freq=2):
        """
        Calculates PMI for all pairs and returns the most 'meaningful' ones.
        """
        results = []
        total = self.total_weighted_contexts
        
        for ((k1, k2), pair_count), ((k1,k2), pair_weight) in zip(self.pair_counts.items(), self.pair_weights.items()):
            if pair_count < min_freq or pair_weight < min_weight:
                continue
                
            count_k1 = self.single_counts[k1]
            count_k2 = self.single_counts[k2]

            weight_k1 = self.single_weights[k1]
            weight_k2 = self.single_weights[k2]
            
            # p_x = count_k1 / total
            # p_y = count_k2 / total
            # p_xy = pair_count / total
            p_x = weight_k1 / total
            p_y = weight_k2 / total
            p_xy = pair_weight / total
            
            # PMI Formula: log( P(x,y) / (P(x)*P(y)) )
            # We add a tiny epsilon to avoid division by zero
            try:
                lift = p_xy / (p_x * p_y)
                pmi = math.log2(lift)
            except:
                pmi = 0.0
            
            if pmi >= min_pmi:
                results.append({
                    "pair": f"{k1} -- {k2}",
                    "freq": pair_count,
                    "weighted_freq": round(pair_weight, 4),
                    "PMI": round(pmi, 3),
                    "strength": round(sigmoid(pmi), 3),
                    "confidence": round(p_xy, 4),
                    "Lift": round(lift, 2)
                })
        
        return sorted(results, key=lambda x: x['PMI'], reverse=True)

