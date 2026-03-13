import unittest
import tempfile
import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Representation.file_parser import parse_file

class TestFileParser(unittest.TestCase):
    """
    Tests for parse_file in file_parser.py
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix='fp_test_')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_file(self, content: str, filename: str) -> str:
        path = os.path.join(self.test_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_csv_format(self):
        """Test parsing a standard CSV file."""
        content = "A,B,O\n1,0,1\n0,1,0"
        path = self._create_file(content, "test.csv")
        rows, targets = parse_file(path, output_col='O')
        
        self.assertEqual(len(rows), 2)
        # 1 -> True, 0 -> False due to _parse_value logic
        self.assertEqual(rows[0], {'A': True, 'B': False})
        self.assertEqual(targets, [True, False])

    def test_txt_format_tab(self):
        """Test parsing a TXT file with tab delimiter (default for non-csv)."""
        # Note: Using literal tabs in the string
        content = "A\tB\tO\n1\t0\t1\n0\t1\t0"
        path = self._create_file(content, "test.txt")
        # For .txt files, parse_file defaults to \t delimiter
        rows, targets = parse_file(path, output_col='O')
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {'A': True, 'B': False})
        self.assertEqual(targets, [True, False])

    def test_custom_delimiter(self):
        """Test parsing with a custom delimiter."""
        content = "A|B|O\n1|0|1\n0|1|0"
        path = self._create_file(content, "test.dat")
        rows, targets = parse_file(path, delimiter='|', output_col='O')
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {'A': True, 'B': False})
        self.assertEqual(targets, [True, False])

    def test_boolean_parsing(self):
        """Test boolean value parsing logic."""
        content = "A,B,O\nTRUE,FALSE,YES\n1,0,NO"
        path = self._create_file(content, "bools.csv")
        rows, targets = parse_file(path, output_col='O')
        
        self.assertEqual(rows[0], {'A': True, 'B': False})
        self.assertEqual(targets[0], True) # YES -> True
        self.assertEqual(rows[1], {'A': True, 'B': False}) # 1 -> True, 0 -> False
        self.assertEqual(targets[1], False) # NO -> False

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        path = os.path.join(self.test_dir, "nonexistent.csv")
        rows, targets = parse_file(path)
        self.assertEqual(rows, [])
        self.assertEqual(targets, [])

if __name__ == '__main__':
    unittest.main(verbosity=2)
