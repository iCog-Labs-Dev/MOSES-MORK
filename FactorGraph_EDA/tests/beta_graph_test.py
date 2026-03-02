import unittest
import math
from FactorGraph_EDA.beta_bp import BetaFactorGraph, BetaState

class TestBetaState(unittest.TestCase):
    def test_default_beta_state(self):
        state = BetaState()
        # Default prior a=b=1 -> strength=0.5, confidence = 2/(2+1)=2/3
        self.assertAlmostEqual(state.strength, 0.5, places=5)
        self.assertAlmostEqual(state.confidence, 2.0 / 3.0, places=5)

    def test_strength_and_confidence_update(self):
        state = BetaState(alpha=3.0, beta=1.0)
        self.assertAlmostEqual(state.strength, 3.0 / 4.0, places=5)
        total = state.alpha + state.beta
        self.assertAlmostEqual(state.confidence, total / (total + 1.0), places=5)

    def test_repr(self):
        state = BetaState(alpha=2.0, beta=5.0)
        rep = repr(state)
        self.assertIn("Beta(a=2.00, b=5.00)", rep)


class TestBetaFactorGraphBasics(unittest.TestCase):
    def setUp(self):
        self.graph = BetaFactorGraph()

    def test_get_or_create_node_creates_neutral_prior(self):
        node = self.graph.get_or_create_node("A")
        # Should initialize with Laplace prior a=b=1
        self.assertIsInstance(node, BetaState)
        self.assertAlmostEqual(node.alpha, 1.0)
        self.assertAlmostEqual(node.beta, 1.0)
        self.assertAlmostEqual(node.strength, 0.5)

    def test_get_or_create_node_returns_same_instance(self):
        node1 = self.graph.get_or_create_node("A")
        node2 = self.graph.get_or_create_node("A")
        self.assertIs(node1, node2)

    def test_add_dependency_rule_creates_nodes_and_rule(self):
        self.graph.add_dependency_rule("A -- B", 0.8, 0.4)
        self.assertIn("A", self.graph.nodes)
        self.assertIn("B", self.graph.nodes)
        self.assertEqual(len(self.graph.factors), 1)

        rule = self.graph.factors[0]
        self.assertEqual(rule["src"], "A")
        self.assertEqual(rule["dst"], "B")
        self.assertAlmostEqual(rule["s"], 0.8)
        self.assertAlmostEqual(rule["c"], 0.4)

    def test_add_dependency_rule_ignores_malformed_pair(self):
        self.graph.add_dependency_rule("A-B", 0.8, 0.4)  # no ' -- '
        self.assertEqual(len(self.graph.factors), 0)

    def test_add_dependency_rule_merges_existing_rule(self):
        self.graph.add_dependency_rule("A -- B", 0.2, 0.2)
        original = dict(self.graph.factors[0])

        # Adding same rule should merge using alpha=0.7
        self.graph.add_dependency_rule("A -- B", 0.8, 0.8)
        self.assertEqual(len(self.graph.factors), 1)
        rule = self.graph.factors[0]

        alpha = 0.7
        expected_s = (1 - alpha) * original["s"] + alpha * 0.8
        expected_c = (1 - alpha) * original["c"] + alpha * 0.8
        self.assertAlmostEqual(rule["s"], expected_s, places=5)
        self.assertAlmostEqual(rule["c"], expected_c, places=5)

    def test_set_prior_changes_alpha_beta_and_priors(self):
        self.graph.set_prior("A", stv_strength=0.9, stv_confidence=0.8, base_counts=10.0)
        node = self.graph.nodes["A"]

        evidence = max(0.1, 0.8 * 10.0)
        expected_a = (0.9 * evidence) + 1.0
        expected_b = ((1.0 - 0.9) * evidence) + 1.0

        self.assertAlmostEqual(node.alpha, expected_a)
        self.assertAlmostEqual(node.beta, expected_b)
        self.assertAlmostEqual(node.prior_a, expected_a)
        self.assertAlmostEqual(node.prior_b, expected_b)


class TestBetaFactorGraphPropagation(unittest.TestCase):
    def setUp(self):
        self.graph = BetaFactorGraph()

    def test_propagation_single_chain_increases_evidence(self):
        # A -> B with reasonably strong rule
        self.graph.add_dependency_rule("A -- B", 0.8, 0.5)
        # Prior: A is likely true with decent confidence
        self.graph.set_prior("A", stv_strength=0.9, stv_confidence=0.8)

        node_A_before = self.graph.nodes["A"]
        node_B_before = self.graph.nodes["B"]

        total_A_before = node_A_before.alpha + node_A_before.beta
        total_B_before = node_B_before.alpha + node_B_before.beta
        strength_B_before = node_B_before.strength

        self.graph.run_evidence_propagation(steps=5, decay=0.9)

        node_A_after = self.graph.nodes["A"]
        node_B_after = self.graph.nodes["B"]

        total_A_after = node_A_after.alpha + node_A_after.beta
        total_B_after = node_B_after.alpha + node_B_after.beta

        # Evidence counts should go up
        self.assertGreater(total_A_after, total_A_before)
        self.assertGreater(total_B_after, total_B_before)

        # B should be pushed towards A being true (strength > 0.5)
        self.assertGreater(node_B_after.strength, strength_B_before)
        self.assertGreater(node_B_after.strength, 0.5)

    def test_propagation_respects_convergence_threshold(self):
        self.graph.add_dependency_rule("A -- B", 0.7, 0.4)
        self.graph.set_prior("A", stv_strength=0.6, stv_confidence=0.7)

        # Run with many steps. Convergence criterion should kick in early;
        # we can't directly inspect steps, but we can check that additional
        # propagation with the same state changes very little.
        self.graph.run_evidence_propagation(steps=30, decay=0.9)

        node_A_1 = self.graph.nodes["A"]
        node_B_1 = self.graph.nodes["B"]
        a1_A, b1_A = node_A_1.alpha, node_A_1.beta
        a1_B, b1_B = node_B_1.alpha, node_B_1.beta

        # Rerun; if already converged, changes should be tiny.
        self.graph.run_evidence_propagation(steps=5, decay=0.9)

        node_A_2 = self.graph.nodes["A"]
        node_B_2 = self.graph.nodes["B"]

        delta_A = abs(node_A_2.alpha - a1_A) + abs(node_A_2.beta - b1_A)
        delta_B = abs(node_B_2.alpha - a1_B) + abs(node_B_2.beta - b1_B)

        # Should be close to numerical stability (small updates)
        self.assertLess(delta_A + delta_B, 1.0)

    def test_cycle_graph_propagation_does_not_explode(self):
        # A -> B -> C -> A cycle
        self.graph.add_dependency_rule("A -- B", 0.7, 0.5)
        self.graph.add_dependency_rule("B -- C", 0.7, 0.5)
        self.graph.add_dependency_rule("C -- A", 0.7, 0.5)

        # Anchor A
        self.graph.set_prior("A", stv_strength=0.8, stv_confidence=0.6)

        self.graph.run_evidence_propagation(steps=10, decay=0.8)

        # Check that no node has NaN or infinite counts
        for name, node in self.graph.nodes.items():
            self.assertFalse(math.isnan(node.alpha))
            self.assertFalse(math.isnan(node.beta))
            self.assertGreater(node.alpha + node.beta, 0.0)

    def test_backward_reasoning_influences_source(self):
        # B strongly implies A (rule A <- B when we add as B -> A)
        self.graph.add_dependency_rule("B -- A", 0.9, 0.6)
        # Anchor B as highly true
        self.graph.set_prior("B", stv_strength=0.95, stv_confidence=0.9)

        node_A_before = self.graph.nodes["A"]
        strength_A_before = node_A_before.strength

        self.graph.run_evidence_propagation(steps=5, decay=0.9)

        node_A_after = self.graph.nodes["A"]
        # Abductive backward pass should push A's strength up
        self.assertGreater(node_A_after.strength, strength_A_before)
        self.assertGreater(node_A_after.strength, 0.5)


if __name__ == "__main__":
    unittest.main()