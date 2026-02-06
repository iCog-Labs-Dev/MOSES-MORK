from Representation.helpers import (TreeNode, tokenize, parse_sexpr, 
                                    add_arg, replace_one_symbol, 
                                    exclude_one_symbol, isOP)
import unittest

class TestTreeNode(unittest.TestCase):
    def test_tree_node_creation(self):
        node = TreeNode("AND")
        self.assertEqual(node.label, "AND")
        self.assertEqual(len(node.children), 0)
        self.assertTrue(node.is_leaf())

    def test_add_child(self):
        parent = TreeNode("AND")
        child1 = TreeNode("A")
        child2 = TreeNode("B")
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        self.assertEqual(len(parent.children), 2)
        self.assertFalse(parent.is_leaf())
        self.assertEqual(parent.children[0].label, "A")
        self.assertEqual(parent.children[1].label, "B")

    def test_repr_leaf_node(self):
        node = TreeNode("A")
        self.assertEqual(str(node), "A")

    def test_repr_parent_node(self):
        parent = TreeNode("AND")
        parent.add_child(TreeNode("A"))
        parent.add_child(TreeNode("B"))
        self.assertEqual(str(parent), "(AND A B)")

class TestTreeParsing(unittest.TestCase):
    def test_tokenize_simple(self):
        s = "(AND A B C)"
        tokens = tokenize(s)
        self.assertEqual(tokens, ["(", "AND", "A", "B", "C", ")"])

    def test_tokenize_empty_string(self):
        s = ""
        tokens = tokenize(s)
        self.assertEqual(tokens, [])

    def test_tokenize_single_atom(self):
        s = "A"
        tokens = tokenize(s)
        self.assertEqual(tokens, ["A"])

    def test_tokenize_nested(self):
        s = "(AND (NOT A) B)"
        tokens = tokenize(s)
        self.assertEqual(tokens, ["(", "AND", "(", "NOT", "A", ")", "B", ")"])

    def test_parse_simple_and(self):
        s = "(AND A B C)"
        tokens = tokenize(s)
        root = parse_sexpr(tokens)

        self.assertEqual(root.label, "AND")
        self.assertEqual(len(root.children), 3)
        self.assertTrue(all(child.is_leaf() for child in root.children))
        self.assertEqual(str(root), "(AND A B C)")

    def test_parse_single_atom(self):
        s = "A"
        tokens = tokenize(s)
        root = parse_sexpr(tokens)
        
        self.assertEqual(root.label, "A")
        self.assertTrue(root.is_leaf())
        self.assertEqual(str(root), "A")

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

    def test_parse_empty_tokens_error(self):
        with self.assertRaises(ValueError):
            parse_sexpr([])

    def test_parse_unexpected_closing_paren(self):
        tokens = [")"]
        with self.assertRaises(ValueError):
            parse_sexpr(tokens)

class TestAddArg(unittest.TestCase):
    def test_add_arg_with_placeholder(self):
        expr = "(AND $ B)"
        result = add_arg(expr, "A")
        self.assertEqual(result, "(AND A B)")

    def test_add_arg_multiple_placeholders(self):
        expr = "(AND $ $ C)"
        result = add_arg(expr, "A")
        self.assertEqual(result, "(AND A $ C)")  # Only replaces first occurrence

    def test_add_arg_no_placeholder(self):
        expr = "(AND A B)"
        result = add_arg(expr, "C")
        self.assertEqual(result, "(AND A B C)")

    def test_add_arg_invalid_expression(self):
        expr = "(AND A B"
        with self.assertRaises(ValueError):
            add_arg(expr, "C")

    def test_add_arg_whitespace_handling(self):
        expr = "  (AND A B)  "
        result = add_arg(expr, "C")
        self.assertEqual(result, "(AND A B C)")

class TestReplaceOneSymbol(unittest.TestCase):
    def test_replace_simple(self):
        expr = "(AND A B C)"
        result = replace_one_symbol(expr, "A", "X")
        self.assertEqual(result, "(AND X B C)")

    def test_replace_first_occurrence_only(self):
        expr = "(AND A B A)"
        result = replace_one_symbol(expr, "A", "X")
        self.assertEqual(result, "(AND X B A)")

    def test_replace_nested_expression(self):
        expr = "(AND (NOT A) B)"
        result = replace_one_symbol(expr, "A", "X")
        self.assertEqual(result, "(AND (NOT X) B)")

    def test_replace_nonexistent_symbol(self):
        expr = "(AND A B C)"
        result = replace_one_symbol(expr, "Z", "X")
        self.assertEqual(result, "(AND A B C)")  # No change

    def test_replace_operator(self):
        expr = "(AND A B)"
        result = replace_one_symbol(expr, "AND", "OR")
        self.assertEqual(result, "(OR A B)")

class TestExcludeOneSymbol(unittest.TestCase):
    def test_exclude_simple(self):
        expr = "(AND A B C)"
        result = exclude_one_symbol(expr, "B")
        self.assertEqual(result, "(AND A C)")

    def test_exclude_first_occurrence_only(self):
        expr = "(AND A B A)"
        result = exclude_one_symbol(expr, "A")
        self.assertEqual(result, "(AND B A)")

    def test_exclude_nested_expression(self):
        expr = "(AND (NOT A) B C)"
        result = exclude_one_symbol(expr, "A")
        self.assertEqual(result, "(AND (NOT) B C)")

    def test_exclude_nonexistent_symbol(self):
        expr = "(AND A B C)"
        result = exclude_one_symbol(expr, "Z")
        self.assertEqual(result, "(AND A B C)")  # No change

    def test_exclude_operator(self):
        expr = "(AND A B)"
        result = exclude_one_symbol(expr, "AND")
        self.assertEqual(result, "(A B)")

class TestIsOP(unittest.TestCase):
    def test_is_op_and(self):
        self.assertTrue(isOP("AND"))

    def test_is_op_or(self):
        self.assertTrue(isOP("OR"))

    def test_is_op_not(self):
        self.assertTrue(isOP("NOT"))

    def test_is_op_variable(self):
        self.assertFalse(isOP("A"))
        self.assertFalse(isOP("B"))
        self.assertFalse(isOP("X"))

    def test_is_op_lowercase(self):
        self.assertFalse(isOP("and"))
        self.assertFalse(isOP("or"))
        self.assertFalse(isOP("not"))

    def test_is_op_empty_string(self):
        self.assertFalse(isOP(""))

    def test_is_op_other_symbols(self):
        self.assertFalse(isOP("("))
        self.assertFalse(isOP(")"))
        self.assertFalse(isOP("$"))

if __name__ == "__main__":
    unittest.main()
