import unittest
import random
from Representation.representation import (
    knobs_from_truth_table,
    initialize_deme,
    sample_random_instances,
    sample_uniform_random_instances,
    # select_top_k,
    build_factor_graph_from_deme,
    Instance,
    Knob,
    Hyperparams,
    Deme
)

class TestExp(unittest.TestCase):

    def setUp(self):
        self.ITable = [
            {"A": True,  "B": True},
            {"A": True,  "B": False},
            {"A": False, "B": True},
            {"A": False, "B": False},
        ]
        self.sketch = "(AND $ $)"

    def test_knobs_from_truth_table(self):
        knobs = knobs_from_truth_table(self.ITable)
        self.assertEqual(len(knobs), 2)
        symbols = [k.symbol for k in knobs]
        self.assertIn("A", symbols)
        self.assertIn("B", symbols)
        a_knob = next(k for k in knobs if k.symbol == "A")
        self.assertEqual(a_knob.Value, [True, True, False, False])

    def test_initialize_deme(self):
        deme = initialize_deme(self.sketch, self.ITable)
        self.assertIsInstance(deme, Deme)
        self.assertGreaterEqual(len(deme.instances), 2)
        
        inst0 = deme.instances[0]
        self.assertTrue(inst0.value.startswith("(AND "))
        self.assertFalse("$" in inst0.value)

    def test_sample_random_instances(self):
        knobs = [Knob("X", 1, [True, False])]
        parent = Instance(value="(NOT X)", id=1, score=0.0, knobs=knobs)
        hyper = Hyperparams(mutation_rate=1.1, crossover_rate=0.0)
        
        child = sample_random_instances(parent, hyper)
        self.assertNotEqual(child.id, parent.id)
        self.assertIn("(NOT X)", child.value)

    def test_sample_uniform_random_instances(self):
        knobs=knobs_from_truth_table(self.ITable)
        random.seed(42)
        inst=sample_uniform_random_instances(self.sketch,knobs,instance_id=42)
        self.assertEqual(inst.id,42)
        self.assertEqual(inst.value , "(AND A)")
       
        self.assertFalse("$" in inst.value)



    # def test_build_factor_graph(self):
    #     i1 = Instance(value="(AND A B)", id=1, score=0.0, knobs=[Knob("A", 1, []), Knob("B", 2, [])])
    #     i2 = Instance(value="(AND A C)", id=2, score=0.0, knobs=[Knob("A", 1, []), Knob("C", 3, [])])
    #     deme = Deme([i1, i2], "fg_test", Hyperparams(0.1, 0.1))
        
    #     fg = build_factor_graph_from_deme(deme)
    #     self.assertEqual(len(fg.variables), 2)
    #     self.assertEqual(len(fg.factors), 1)
    #     factor = fg.factors[0]
    #     self.assertIn(i1, factor.variables)
    #     self.assertIn(i2, factor.variables)
    #     score = factor.evaluate()
    #     self.assertEqual(score, 2.0)

if __name__ == '__main__':
    unittest.main()