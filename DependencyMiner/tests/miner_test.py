import unittest

from DependencyMiner.miner import (
    TreeNode,
    tokenize,
    parse_sexpr,
    OrderedTreeMiner,
    DependencyMiner,
)


class TestTreeParsing(unittest.TestCase):
    def test_tokenize_simple(self):
        s = "(AND A B C)"
        tokens = tokenize(s)
        self.assertEqual(tokens, ["(", "AND", "A", "B", "C", ")"])

    def test_parse_simple_and(self):
        s = "(AND A B C)"
        tokens = tokenize(s)
        root = parse_sexpr(tokens)

        self.assertEqual(root.label, "AND")
        self.assertEqual(len(root.children), 3)
        self.assertTrue(all(child.is_leaf() for child in root.children))
        self.assertEqual(str(root), "(AND A B C)")

    def test_parse_nested_or_not(self):
        s = "(AND (NOT A) (OR (NOT B) C))"
        tokens = tokenize(s)
        root = parse_sexpr(tokens)

        self.assertEqual(root.label, "AND")
        self.assertEqual(len(root.children), 2)

        not_node = root.children[0]
        or_node = root.children[1]

        self.assertEqual(not_node.label, "NOT")
        self.assertEqual(len(not_node.children), 1)
        self.assertEqual(not_node.children[0].label, "A")

        self.assertEqual(or_node.label, "OR")
        self.assertEqual(len(or_node.children), 2)
        self.assertEqual(str(root), "(AND (NOT A) (OR (NOT B) C))")

    def test_parse_grouping(self):
        # ((NOT A) B) should introduce an implicit GROUP node
        s = "((NOT A) B)"
        tokens = tokenize(s)
        root = parse_sexpr(tokens)

        self.assertEqual(root.label, "GROUP")
        self.assertEqual(len(root.children), 2)
        self.assertEqual(str(root.children[0]), "(NOT A)")
        self.assertEqual(str(root.children[1]), "B")


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

    def test_fit_and_counts(self):
        miner = DependencyMiner().fit(self.data)

        # There must be some contexts
        self.assertGreater(miner.total_contexts, 0)

        # At least A, B, C (or their NOT/OR variants) must appear as single keys
        single_keys = set(miner.single_counts.keys())
        self.assertTrue(any("A" in k for k in single_keys))
        self.assertTrue(any("B" in k for k in single_keys))
        self.assertTrue(any("C" in k for k in single_keys))

        # There should be some sibling pairs
        self.assertGreater(len(miner.pair_counts), 0)

    def test_meaningful_dependencies(self):
        miner = DependencyMiner().fit(self.data)
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


if __name__ == "__main__":
    unittest.main()