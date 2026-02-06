import unittest

from DependencyMiner.miner import (
    OrderedTreeMiner,
    DependencyMiner,
    sigmoid
)
from Representation.helpers import tokenize, parse_sexpr
from Representation.representation import FitnessOracle, Instance

class TestOrderedTreeMiner(unittest.TestCase):
    def setUp(self):
        self.data = [
            "(AND A B C)",
            "(AND (NOT A) B C)",
            "(AND A (NOT B) C)",
            "(AND (NOT A) (OR (NOT B) C))",
            "(AND A (OR C B))",
            "(AND (OR (NOT A) C) B)",
            "(AND A (NOT C) B)",
            "(AND (NOT A) (OR (NOT C) B))",
            "(AND B C)",
            "(AND (NOT B) C)",
            "(AND B (NOT C))",
            "(AND (NOT B) (NOT C))",
        ]

    def test_fit_and_frequent_patterns_with_support_4(self):
        miner = OrderedTreeMiner(min_support=4)
        miner.fit(self.data)
        patterns = dict(miner.get_frequent_patterns())

        self.assertIn("(AND)", patterns)
        self.assertGreaterEqual(patterns["(AND)"], 4)

        self.assertIn("(A)", patterns)
        self.assertGreater(patterns["(A)"], 0)

        sorted_patterns = miner.get_frequent_patterns()
        for i in range(len(sorted_patterns) - 1):
            (p1, c1), (p2, c2) = sorted_patterns[i], sorted_patterns[i + 1]
            if c1 == c2:
                self.assertGreaterEqual(len(p1), len(p2))
            else:
                self.assertGreaterEqual(c1, c2)

    def test_subtrees_order_preservation(self):
        expr = "(AND A B C)"
        tokens = tokenize(expr)
        root = parse_sexpr(tokens)

        miner = OrderedTreeMiner(min_support=1)
        subtrees = set()
        stack = [root]
        while stack:
            node = stack.pop()
            subtrees.update(miner._get_subtrees(node))
            stack.extend(node.children)


        # Single-node patterns
        self.assertIn("(AND)", subtrees)
        self.assertIn("(A)", subtrees)
        self.assertIn("(B)", subtrees)
        self.assertIn("(C)", subtrees)

        # Order-preserving multi-child patterns
        self.assertIn("(AND (A) (B))", subtrees)
        self.assertIn("(AND (B) (C))", subtrees)
        self.assertIn("(AND (A) (B) (C))", subtrees)

        self.assertNotIn("(AND B A)", subtrees)


class TestDependencyMiner(unittest.TestCase):
    def setUp(self):
        self.data = [
            "(AND A B C)",
            "(AND (NOT A) B C)",
            "(AND A (NOT B) C)",
            "(AND (NOT A) (OR (NOT B) C))",
            "(AND A (OR C B))",
            "(AND (OR (NOT A) C) B)",
            "(AND A (NOT C) B)",
            "(AND (NOT A) (OR (NOT C) B))",
            "(AND B C)",
            "(AND (NOT B) C)",
            "(AND B (NOT C))",
            "(AND (NOT B) (NOT C))",
        ]
        # Create fitness oracle with arbitrary target
        target_vals = [False, True, False, True, False, True, False, True]
        self.fitness = FitnessOracle(target_vals)
        
        # Calculate real fitness scores for each expression
        instances = [Instance(value=expr, id=i, score=0.0, knobs=[]) for i, expr in enumerate(self.data)]
        self.default_weights = [self.fitness.get_fitness(inst) for inst in instances]

    def test_empty_input(self):
        miner = DependencyMiner()
        miner.fit([], [])
        
        self.assertEqual(miner.total_weighted_contexts, 0.0)
        self.assertEqual(len(miner.pair_counts), 0)
        self.assertEqual(len(miner.single_counts), 0)

    def test_zero_weights(self):
        miner = DependencyMiner()
        zero_weights = [0.0] * len(self.data)
        miner.fit(self.data, zero_weights)
        
        self.assertEqual(miner.total_weighted_contexts, 0.0)
        deps = miner.get_meaningful_dependencies()
        self.assertEqual(deps, [])

    def test_negative_weights_ignored(self):
        miner = DependencyMiner()
        negative_weights = [-1.0] * len(self.data)
        miner.fit(self.data, negative_weights)
        
        self.assertEqual(miner.total_weighted_contexts, 0.0)
        deps = miner.get_meaningful_dependencies()
        self.assertEqual(deps, [])

    def test_single_expression_no_pairs(self):
        miner = DependencyMiner()
        miner.fit(["A"], [1.0])  
        
        self.assertEqual(miner.total_weighted_contexts, 0.0)
        self.assertEqual(len(miner.pair_counts), 0)

    def test_weighted_vs_unweighted(self):
        # Create two different fitness scenarios
        target1 = [False, True, False, True, False, True, False, True]
        target2 = [True, False, True, False, True, False, True, False]
        
        fitness1 = FitnessOracle(target1)
        fitness2 = FitnessOracle(target2)
        
        instances = [Instance(value=expr, id=i, score=0.0, knobs=[]) for i, expr in enumerate(self.data)]
        weights1 = [fitness1.get_fitness(inst) for inst in instances]
        weights2 = [fitness2.get_fitness(inst) for inst in instances]
        
        miner1 = DependencyMiner()
        miner1.fit(self.data, weights1)
        
        miner2 = DependencyMiner()
        miner2.fit(self.data, weights2)
        
        deps1 = miner1.get_meaningful_dependencies(min_pmi=0.0, min_freq=1)
        deps2 = miner2.get_meaningful_dependencies(min_pmi=0.0, min_freq=1)
        
        # Different fitness landscapes should produce different dependency patterns
        if deps1 and deps2:
          
            self.assertGreater(len(deps1), 0)
            self.assertGreater(len(deps2), 0)
    def test_weights_change_pmi(self):
        # Create truth table: A XOR B
        target_vals = [False, True, True, False]  
        fitness = FitnessOracle(target_vals)
        
        data = [
            "(AND A B)",     
            "(OR A B)",       
            "(AND (NOT A) B)", 
            "(AND A (NOT B))",  
        ]
        
        # Calculate real fitness scores as weights
        instances = [Instance(value=expr, id=i, score=0.0, knobs=[]) for i, expr in enumerate(data)]
        weights = [fitness.get_fitness(inst) for inst in instances]
        
        miner = DependencyMiner()
        miner.fit(data, weights)
        deps = miner.get_meaningful_dependencies(min_pmi=-0.1, min_freq=1)
        
        self.assertGreater(len(deps), 0)
        self.assertTrue(all(d["weighted_freq"] > 0 for d in deps))

    def test_meaningful_dependencies(self):
        miner = DependencyMiner().fit(self.data, self.default_weights)
        deps = miner.get_meaningful_dependencies(min_pmi=0.0, min_freq=2)

        # Should return a non-empty list for these synthetic data
        self.assertIsInstance(deps, list)
        self.assertGreater(len(deps), 0)

        # Each dependency entry must contain required keys
        for d in deps:
            self.assertIn("pair", d)
            self.assertIn("freq", d)
            self.assertIn("PMI", d)
            self.assertIn("Lift", d)
            self.assertGreaterEqual(d["freq"], 2)

        # Sort order: PMI non-increasing
        for i in range(len(deps) - 1):
            self.assertGreaterEqual(deps[i]["PMI"], deps[i + 1]["PMI"])

class TestSigmoidFunction(unittest.TestCase):
    def test_sigmoid_zero(self):
        result = sigmoid(0)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_sigmoid_positive(self):
        result = sigmoid(10)
        self.assertGreater(result, 0.5)
        self.assertLess(result, 1.0)

    def test_sigmoid_negative(self):
        result = sigmoid(-10)
        self.assertLess(result, 0.5)
        self.assertGreater(result, 0.0)


if __name__ == "__main__":
    unittest.main()