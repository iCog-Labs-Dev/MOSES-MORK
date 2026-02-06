import unittest
import random
from Variation_quantale.mutation import Mutation
from Representation.representation import Instance, Hyperparams, Knob
from Representation.helpers import tokenize

class MutationTestCase(unittest.TestCase):
    def setUp(self):
        random.seed(42)

        self.knobs = [
            Knob(symbol="A", id=1, Value=[True, False]),
            Knob(symbol="B", id=2, Value=[True, True]),
            Knob(symbol="C", id=3, Value=[False, True]),
            Knob(symbol="D", id=4, Value=[True, False]),
            Knob(symbol="E", id=5, Value=[False, False]),
            Knob(symbol="F", id=6, Value=[True, True]),
        ]

        self.parent_expr = "(AND A B C (OR D (AND E F)))"
        self.parent = Instance(
            value=self.parent_expr,
            id=0,
            score=0.0,
            knobs=self.knobs,
        )

        self.stv = {
            "A": (0.95, 0.95),  # Strong
            "B": (0.80, 0.80),  # Good
            "C": (0.20, 0.20),  # Weak
            "D": (0.35, 0.50),  # Weak
            "E": (0.46, 0.50),  # Weak
            "F": (0.46, 0.50),  # Weak
            "(OR D (AND E F))": (0.7, 0.5),  # Moderate
        }

        self.hyper = Hyperparams(
            mutation_rate=0.3,
            crossover_rate=0.7,
            neighborhood_size=10,
            num_generations=10,
        )

        self.mutator = Mutation(self.parent, self.stv, self.hyper)


    def test_get_base_OP(self):
        self.assertEqual(self.mutator.base_op, "AND")

    def test_reference_order_parsed(self):
        self.assertIn("A", self.mutator.reference_order)
        self.assertIn("B", self.mutator.reference_order)
        self.assertIn("C", self.mutator.reference_order)
        self.assertIn("(OR D (AND E F))", self.mutator.reference_order)
        self.assertEqual(len(self.mutator.reference_order), 4)

    # ---------- join (NOT flipping) ----------

    def test_join_negates_symbol(self):
        self.assertEqual(self.mutator.join("C"), "(NOT C)")

    def test_join_unnests_negated_symbol(self):
        self.assertEqual(self.mutator.join("(NOT C)"), "C")

    # ---------- composite score ----------

    def test_get_composite_score_atomic(self):
        score_A = self.mutator._get_composite_score("A")
        self.assertAlmostEqual(score_A, (0.95 + 0.95) / 2.0)

    def test_get_composite_score_complex_uses_children(self):
        score_complex = self.mutator._get_composite_score("(OR D (AND E F))")
        self.assertAlmostEqual(score_complex, (0.7 + 0.5) / 2.0)

    def test_get_composite_score_fallback_mutation_rate(self):
        score_unknown = self.mutator._get_composite_score("(AND X Y)")
        self.assertAlmostEqual(score_unknown, self.hyper.mutation_rate)

    # ---------- multiplicative mutation ----------

    def test_execute_multiplicative_returns_instance(self):
        child = self.mutator.execute_multiplicative()
        self.assertIsInstance(child, Instance)
        self.assertIsInstance(child.value, str)

    def test_execute_multiplicative_operator_preserved(self):
        child = self.mutator.execute_multiplicative()
        self.assertTrue(child.value.startswith("(AND"))
        self.assertTrue(child.value.endswith(")"))

    def test_execute_multiplicative_knobs_subset_of_parent(self):
        child = self.mutator.execute_multiplicative()
        parent_symbols = {k.symbol for k in self.parent.knobs}
        child_symbols = {k.symbol for k in child.knobs}
        self.assertTrue(child_symbols.issubset(parent_symbols))

    def test_execute_multiplicative_empty_kept_uses_base_op_only(self):
        hyper_low = Hyperparams(
            mutation_rate=0.0001,
            crossover_rate=0.7,
            neighborhood_size=10,
            num_generations=10,
        )
        stv_zero = {k: (0.0, 0.0) for k in tokenize(self.parent_expr)}
        mutator_low = Mutation(self.parent, stv_zero, hyper_low)

        # product() is stochastic, but with such low rate and zero scores,
        # it's extremely likely all will be pruned; we assert the fallback format
        child = mutator_low.execute_multiplicative()
        self.assertEqual(child, f"({mutator_low.base_op})")

    # ---------- additive mutation ----------

    def test_execute_additive_returns_instance(self):
        child = self.mutator.execute_additive()
        self.assertIsInstance(child, Instance)

    def test_execute_additive_operator_preserved(self):
        child = self.mutator.execute_additive()
        self.assertTrue(child.value.startswith("(AND"))
        self.assertTrue(child.value.endswith(")"))

    def test_execute_additive_knobs_match_present_symbols(self):
        child = self.mutator.execute_additive()
        present_symbols = set(tokenize(child.value))
        child_symbols = {k.symbol for k in child.knobs}
        self.assertTrue(child_symbols.issubset(present_symbols))

    def test_execute_additive_uses_custom_base_mutation_rate(self):
        child = self.mutator.execute_additive(base_mutation_rate=0.9)
        self.assertIn("(NOT", child.value)

    # ---------- internal _mutate_expression ----------

    def test_mutate_expression_atomic_returns_string(self):
        mutated = self.mutator._mutate_expression("A")
        self.assertIsInstance(mutated, str)

    def test_mutate_expression_complex_preserves_structure(self):
        expr = "(AND A (OR B C))"
        mutated = self.mutator._mutate_expression(expr)
        self.assertTrue(mutated.startswith("("))
        self.assertTrue(mutated.endswith(")"))

    def test_product_on_atomic_symbol(self):
        result = self.mutator.product("A")
        self.assertIn(result, ("A", None))


if __name__ == "__main__":
    unittest.main()