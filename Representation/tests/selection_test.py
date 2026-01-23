import unittest
from Representation.selection import (
    select_top_k,
    tournament_selection,
)
from Representation.representation import (
    Instance,
    Deme,
    Knob,
    Hyperparams
)

class TestSelection(unittest.TestCase):
    def setUp(self):
        self.ITable = [
            {"A": True,  "B": True,  "O": True},
            {"A": True,  "B": False, "O": False},
            {"A": False, "B": True,  "O": False},
            {"A": False, "B": False, "O": False},
        ]
        self.sketch = "(AND $ $)"

    def test_select_top_k(self):
            
            kA = Knob(symbol="A", id=1, Value=[True, False])
            kB = Knob(symbol="B", id=2, Value=[True, False])
            kC = Knob(symbol="C", id=3, Value=[True, False])
            
            i1 = Instance(value="(AND A)", id=1, score=0.9, knobs=[kA])
            i2 = Instance(value="(AND A B)", id=2, score=0.6, knobs=[kA, kB])
            i3 = Instance(value="(AND A B C)", id=3, score=0.4, knobs=[kA, kB, kC])
            deme = Deme([i1, i2, i3], "Deme-01", Hyperparams(0.1, 0.1, 10, 5))
            
            top = select_top_k(deme, k=2)
            self.assertEqual(len(top), 2)
            self.assertEqual(top[0].id, 1)
            self.assertEqual(top[1].id, 2)
    
    def test_tournament_selection(self):
        kA = Knob(symbol="A", id=1, Value=[True, False])
        kB = Knob(symbol="B", id=2, Value=[True, False])
        kC = Knob(symbol="C", id=3, Value=[True, False])
        
        i1 = Instance(value="(AND A)", id=1, score=0.9, knobs=[kA])
        i2 = Instance(value="(AND A B)", id=2, score=0.6, knobs=[kA, kB])
        i3 = Instance(value="(AND A B C)", id=3, score=0.4, knobs=[kA, kB, kC])
        deme = Deme([i1, i2, i3], "Deme-01", Hyperparams(0.1, 0.1, 10, 5))

        selected = tournament_selection(deme, k=2, tournament_size=2)
        self.assertEqual(len(selected), 2)

        winners = tournament_selection(deme, k=1, tournament_size=3)
        self.assertEqual(len(winners), 1)
        self.assertEqual(winners[0].id, 1)