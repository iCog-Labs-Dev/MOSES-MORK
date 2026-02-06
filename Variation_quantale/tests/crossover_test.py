import unittest
import random
from Variation_quantale.crossover import VariationQuantale, get_top_level_features
from Representation.representation import Instance, Knob

class TestGetTopLevelFeatures(unittest.TestCase):
    def test_simple_and(self):
        expr = "(AND A B C)"
        self.assertEqual(get_top_level_features(expr), ["A", "B", "C"])

    def test_nested_or(self):
        expr = "(AND A (OR B C) D)"
        self.assertEqual(get_top_level_features(expr), ["A", "(OR B C)", "D"])

    def test_or_root(self):
        expr = "(OR A (AND B C) D)"
        self.assertEqual(get_top_level_features(expr), ["A", "(AND B C)", "D"])

    def test_whitespace_handling(self):
        expr = " (AND   A   (OR   B   C )   D ) "
        self.assertEqual(get_top_level_features(expr), ["A", "(OR   B   C )", "D"])


class TestVariationQuantale(unittest.TestCase):
    def setUp(self):
            
        self.knob_a = Knob(symbol="A", id=1, Value=[True, False])
        self.knob_b = Knob(symbol="B", id=2, Value=[True, False])
        self.knob_c = Knob(symbol="C", id=3, Value=[True, False])
        self.knob_d = Knob(symbol="D", id=3, Value=[True, False])
        
        self.knobs = [self.knob_a, self.knob_b, self.knob_c, self.knob_d]

        self.m1 = Instance(
            value="(AND A B (OR C D))",
            id=1,
            score=0.0,
            knobs=self.knobs,
        )
        self.m2 = Instance(
            value="(AND B C D)",
            id=2,
            score=0.0,
            knobs=self.knobs,
        )

        # Fixed STV values to make mask probabilities predictable-ish
        self.stv_values = {
            "A": (0.9, 0.9),          
            "B": (0.8, 0.8),          
            "C": (0.2, 0.2),          
            "D": (0.5, 0.5), 
            "(OR C D)": (0.7, 0.7),   
        }

        random.seed(0)
        self.vq = VariationQuantale(self.m1, self.m2, self.stv_values)

    def test_universe_and_reference_order(self):
        # top-level of m1: ["A", "B", "(OR C D)"]
        # top-level of m2: ["B", "C", "D"]
        expected_universe = {"A", "B", "(OR C D)", "C", "D"}
        self.assertEqual(self.vq.universe, expected_universe)

        expected_order = ["A", "B", "(OR C D)", "C", "D"]
        self.assertEqual(self.vq.reference_order, expected_order)

    def test_unit_and_zero(self):
        self.assertEqual(self.vq.unit(), self.vq.universe)
        self.assertEqual(self.vq.zero(), set())

    def test_join_operation(self):
        a = {"A", "B"}
        b = {"B", "C"}
        self.assertEqual(self.vq.join(a, b), {"A", "B", "C"})

    def test_product_operation(self):
        features = {"A", "B", "C"}
        mask = {"B", "C", "D"}
        self.assertEqual(self.vq.product(features, mask), {"B", "C"})

    def test_residium_computes_complement_wrt_unit(self):
        unit = self.vq.unit()
        subset = {"A", "B"}
        comp = self.vq.residium(subset, unit)
        self.assertEqual(comp, unit.difference(subset))

    def test_generate_random_mask_subset_of_universe(self):
        random.seed(1)
        mask = self.vq._generate_random_mask(self.stv_values)
        self.assertTrue(mask.issubset(self.vq.universe))

    def test_execute_crossover_returns_instance(self):
        random.seed(2)
        child = self.vq.execute_crossover()

        # Child must be an Instance
        self.assertIsInstance(child, Instance)

        # Root op should match m1 root ("AND")
        self.assertTrue(child.value.startswith("(AND "))

        # Features must be subset of universe
        features = set(get_top_level_features(child.value))
        self.assertTrue(features.issubset(self.vq.universe))

        # Child knobs should correspond to symbols used
        knob_symbols = {k.symbol for k in child.knobs}
        for f in features:
            # Ignore compound expressions like "(OR C D)"
            if not f.startswith("("):
                self.assertIn(f, knob_symbols)

    def test_execute_crossover_uses_reference_order(self):
        # Fix seed for determinism
        random.seed(3)
        vq = VariationQuantale(self.m1, self.m2, self.stv_values)
        child = vq.execute_crossover()
        child_features = get_top_level_features(child.value)

        # Child features appear in the same relative order as reference_order
        order_positions = {f: i for i, f in enumerate(vq.reference_order)}
        positions = [order_positions[f] for f in child_features]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()