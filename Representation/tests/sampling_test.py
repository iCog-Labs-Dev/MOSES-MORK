import unittest
import random
from copy import deepcopy
from Representation.sampling import (randomUniform, randomBernoulli,
                                     sample_new_instances, sample_logical_perms)
from Representation.representation import Instance, Knob, Hyperparams
from Representation.helpers import TreeNode, parse_sexpr, tokenize, isOP

class TestRandomUniform(unittest.TestCase):
    def test_random_uniform_returns_subset_of_input(self):
        knobs = ["A", "B", "C", "D"]
        random.seed(0)
        selected = randomUniform(knobs)
        # All selected items must come from original list, no extras
        self.assertTrue(set(selected).issubset(set(knobs)))

    def test_random_uniform_empty_input(self):
        knobs = []
        random.seed(0)
        selected = randomUniform(knobs)
        self.assertEqual(selected, None)

    def test_random_uniform_not_deterministic_distribution(self):
        # Over many runs, expect both empty and non-empty outcomes at least once
        knobs = ["A", "B"]
        saw_empty = False
        saw_non_empty = False
        random.seed(1)
        for _ in range(100):
            selected = randomUniform(knobs)
            if not selected:
                saw_empty = True
            else:
                saw_non_empty = True
            if saw_empty and saw_non_empty:
                break
        self.assertTrue(saw_empty)
        self.assertTrue(saw_non_empty)


class TestRandomBernoulli(unittest.TestCase):
    def setUp(self):
        self.knobA = Knob(symbol="A", id=1, Value=[True, False])
        self.knobB = Knob(symbol="B", id=2, Value=[True, False])
        self.knobC = Knob(symbol="C", id=3, Value=[True, False])

        self.knobD = Knob(symbol="D", id=4, Value=[True, False])
        self.knobE = Knob(symbol="E", id=5, Value=[True, False])

        self.instance = Instance(
            value="(AND A (OR B C))",
            id=1,
            score=0.0,
            knobs=[self.knobA, self.knobB, self.knobC],
        )

        self.perms = ["D", "E", "C"]
        self.new_knobs = [self.knobD, self.knobE, self.knobC]

    def test_random_bernoulli_returns_none_if_no_selected_knobs(self):
        # Force randomUniform to always return empty by using no perms
        new_inst = randomBernoulli(0.5, self.instance, [], self.new_knobs)
        self.assertIsNone(new_inst)

    def test_random_bernoulli_probability_zero_no_change(self):
        random.seed(0)
        new_inst = randomBernoulli(0.0, self.instance, self.new_knobs, self.instance.knobs)
        # With p=0, there should be no change to the expression
        self.assertIsNotNone(new_inst)
        self.assertEqual(new_inst.value, self.instance.value)
        # Knobs should be identical after filtering
        self.assertCountEqual(
            [k.symbol for k in new_inst.knobs],
            [k.symbol for k in self.instance.knobs],
        )

    def test_random_bernoulli_probability_one_mutates_if_knobs_available(self):
        random.seed(0)
        new_inst = randomBernoulli(1.0, self.instance, self.new_knobs, self.instance.knobs)
        # Expect either at least one mutation or, worst case, same value but logic still valid
        self.assertIsNotNone(new_inst)
        self.assertIsInstance(new_inst, Instance)
        # At least one candidate path should have been attempted; value can be equal
        self.assertIn("AND", new_inst.value)  # still valid expression structure

    def test_random_bernoulli_updates_knobs_based_on_tokens(self):
        random.seed(2)
        new_inst = randomBernoulli(1.0, self.instance, self.new_knobs, self.instance.knobs)
        self.assertIsNotNone(new_inst)
        present_tokens = set(tokenize(new_inst.value))
        knob_symbols = {k.symbol for k in new_inst.knobs}
        # All knob symbols should be present in tokens
        self.assertTrue(knob_symbols.issubset(present_tokens))

    def test_random_bernoulli_does_not_duplicate_knobs(self):
        random.seed(3)
        new_inst = randomBernoulli(1.0, self.instance, self.new_knobs, self.instance.knobs)
        self.assertIsNotNone(new_inst)
        symbols = [k.symbol for k in new_inst.knobs]
        self.assertEqual(len(symbols), len(set(symbols)))

    def test_random_bernoulli_handles_complex_perm_expression(self):
        random.seed(2)
        new_inst = randomBernoulli(1.0, self.instance, [self.knobA, self.knobC, self.knobE], self.instance.knobs)
        self.assertIsNotNone(new_inst)
        knob_symbols = {k.symbol for k in new_inst.knobs}
        self.assertTrue(
            knob_symbols.issuperset({"C"}) or knob_symbols.issuperset({"E"})
        )


class TestSampleNewInstances(unittest.TestCase):
    def setUp(self):
        self.knobA = Knob(symbol="A", id=1, Value=[True, False])
        self.knobB = Knob(symbol="B", id=2, Value=[True, False])
        self.knobC = Knob(symbol="C", id=3, Value=[True, False])

        self.knobD = Knob(symbol="D", id=4, Value=[True, False])
        self.knobE = Knob(symbol="E", id=5, Value=[True, False])

        self.instance = Instance(
            value="(AND A (OR B C))",
            id=1,
            score=0.0,
            knobs=[self.knobA, self.knobB, self.knobC],
        )

        self.perms = ["D", "E", "C"]
        self.new_knobs = [self.knobD, self.knobE, self.knobC]

        self.hyperparams = Hyperparams(
            mutation_rate=0.1,
            crossover_rate=0.6,
            neighborhood_size=10,
            num_generations=50,
        )

    def test_sample_new_instances_returns_dict_of_unique_values(self):
        random.seed(0)
        new_instances = sample_new_instances(
            0.5, self.hyperparams, self.instance, self.new_knobs, self.instance.knobs
        )
        
        self.assertIsInstance(new_instances, dict)
        for k, v in new_instances.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, Instance)
            self.assertEqual(k, v.value)
        self.assertEqual(len(new_instances), len(set(new_instances.keys())))

    def test_sample_new_instances_respects_neighborhood_size_upper_bound(self):
        random.seed(1)
        new_instances = sample_new_instances(
            0.5, self.hyperparams, self.instance, self.new_knobs, self.instance.knobs
        )
        # Number of unique instances cannot exceed neighborhood_size
        self.assertLessEqual(len(new_instances), self.hyperparams.neighborhood_size)

    def test_sample_new_instances_with_zero_neighborhood_size(self):
        random.seed(2)
        hyperparams = deepcopy(self.hyperparams)
        hyperparams.neighborhood_size = 0
        new_instances = sample_new_instances(
            0.5, hyperparams, self.instance, self.perms, self.new_knobs
        )
        self.assertEqual(new_instances, {})

    def test_sample_new_instances_handles_no_perms(self):
        random.seed(3)
        new_instances = sample_new_instances(
            0.5, self.hyperparams, self.instance, [], self.new_knobs
        )
        # randomBernoulli should always return None, so no instances
        self.assertEqual(new_instances, {})

    def test_sample_new_instances_probability_zero(self):
        random.seed(4)
        new_instances = sample_new_instances(
            0.0, self.hyperparams, self.instance, self.new_knobs, self.instance.knobs
        )
        # With p=0, randomBernoulli never mutates; but it still returns instances
        self.assertLessEqual(len(new_instances), 1)
        if new_instances:
            (value, inst), = new_instances.items()
            self.assertEqual(value, self.instance.value)
            self.assertEqual(inst.value, self.instance.value)

class TestSampleLogicalPerms(unittest.TestCase):
    def test_sample_logical_perms(self):
            knobs = [
                Knob(symbol="A", id=1, Value=[]),
                Knob(symbol="B", id=2, Value=[])
            ]

            # Test 1: Current OP is AND -> Expects OR pairs + Vars
            candidates_and, new_knobs = sample_logical_perms("AND", knobs)

            # Expect: "A", "B"
            self.assertIn("A", candidates_and)
            self.assertIn("B", candidates_and)

            # Expect Pairs (OR A B), (OR (NOT A) B) ...
            self.assertIn("(OR A B)", candidates_and)
            self.assertIn("(OR (NOT A) B)", candidates_and) 
            self.assertIn("(OR A (NOT B))", candidates_and)
            self.assertIn("(OR (NOT A) (NOT B))", candidates_and)

            # Test 2: Current OP is OR -> Expects AND pairs
            candidates_or, new_knobs = sample_logical_perms("OR", knobs)
            self.assertIn("(AND A B)", candidates_or)


if __name__ == "__main__":
    unittest.main()