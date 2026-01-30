import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from Representation.representation import FitnessOracle, Instance, Knob

class TestFitnessOracle(unittest.TestCase):
    def setUp(self):
        # Truth table:
        # A B O (Target: A AND B)
        # 0 0 0
        # 0 1 0
        # 1 0 0
        # 1 1 1
        
        self.knobs = [
            Knob(symbol="A", id=1, Value=[False, False, True, True]),
            Knob(symbol="B", id=2, Value=[False, True, False, True]),
        ]
        
        self.oracle = FitnessOracle([False, False, False, True])

    def create_instance(self, expression):
        return Instance(value=expression, id=1, score=0.0, knobs=self.knobs)

    def test_perfect_match(self):
        inst = self.create_instance("(AND A B)")
        fitness = self.oracle.get_fitness(inst)
        self.assertEqual(fitness, 1.0, "Should be perfect match")
        self.assertEqual(inst.score, 1.0)

    def test_partial_match(self):
        inst = self.create_instance("(OR A B)")
        fitness = self.oracle.get_fitness(inst)
        self.assertEqual(fitness, 0.5, "Should have 0.5 accuracy")

    def test_nested_expression(self):
        inst = self.create_instance("(NOT (AND A B))")
        fitness = self.oracle.get_fitness(inst)
        self.assertEqual(fitness, 0.0)

    def test_complex_nested(self):
        inst = self.create_instance("(OR (AND A B) (NOT A))")
        fitness = self.oracle.get_fitness(inst)
        self.assertEqual(fitness, 0.5)

    def test_caching(self):
        inst = self.create_instance("(AND A B)")
        f1 = self.oracle.get_fitness(inst)
        self.assertIn("(AND A B)", self.oracle.memo)
        
        self.oracle.memo["(AND A B)"] = 0.99
        f2 = self.oracle.get_fitness(inst)
        self.assertEqual(f2, 0.99, "Should return cached value")
        self.assertEqual(inst.score, 0.99)
    
    def test_missing_variables(self):
        inst = self.create_instance("(AND A C)")
        fitness = self.oracle.get_fitness(inst)
        self.assertEqual(fitness, 0.75)

    ## a single variable test 
    def test_single_variable(self):
        inst = self.create_instance("A")
        fitness = self.oracle.get_fitness(inst)
        self.assertEqual(fitness, 0.75)

if __name__ == '__main__':
    unittest.main()
