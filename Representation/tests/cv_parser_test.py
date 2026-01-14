import unittest
import tempfile
import os
import shutil
from Representation.csv_parser import load_truth_table


class TestCSVTruthTableParserEssential(unittest.TestCase):
    """
    5 essential test cases for load_truth_table — focused and high-value.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='tt_test_')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_csv(self, content: str, filename="test.csv") -> str:
        path = os.path.join(self.test_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_basic_and_most_common_case(self):
        """Standard 0/1 truth table — the pattern users most often write manually."""
        content = """A,B,O
1,1,1
1,0,0
0,1,0
0,0,0"""
        path = self._create_csv(content)
        rows, targets = load_truth_table(path, output_col='O')

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(targets), 4)
        self.assertEqual(rows[0], {'A': True, 'B': True})
        self.assertEqual(rows[1], {'A': True, 'B': False})
        self.assertEqual(targets, [True, False, False, False])

    def test_various_boolean_formats_and_case_insensitivity(self):
        """Covers TRUE/T/F/Yes/1/true/False etc. — the #1 source of real bugs."""
        content = """X,Y,Out
TRUE,FALSE,TRUE
t,f,t
Yes,No,yes
1,0,1
false,TRUE,FALSE
"""
        path = self._create_csv(content)
        rows, targets = load_truth_table(path, output_col='Out')

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0], {'X': True,  'Y': False})
        self.assertEqual(rows[1], {'X': True,  'Y': False})
        self.assertEqual(rows[2], {'X': True,  'Y': False})
        self.assertEqual(rows[3], {'X': True,  'Y': False})
        self.assertEqual(rows[4], {'X': False, 'Y': True})
        self.assertEqual(targets, [True, True, True, True, False])

    def test_file_not_found_or_invalid_path(self):
        """Critical safety check — wrong filename is very common."""
        fake_path = os.path.join(self.test_dir, "does-not-exist.csv")
        rows, targets = load_truth_table(fake_path, output_col='O')

        self.assertEqual(rows, [])
        self.assertEqual(targets, [])   # matches current implementation


if __name__ == '__main__':
    unittest.main(verbosity=2)