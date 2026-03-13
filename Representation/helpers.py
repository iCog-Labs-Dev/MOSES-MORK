from typing import List, Union
import re


class TreeNode:
    def __init__(self, label):
        self.label = label
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def is_leaf(self):
        return len(self.children) == 0

    def __repr__(self):
        # Canonical string representation (DFS Pre-order)
        if self.is_leaf():
            return self.label
        child_strs = " ".join([str(c) for c in self.children])
        return f"({self.label} {child_strs})"


LOGIC_OPERATORS = {"AND", "OR", "NOT"}

def tokenize(s_expr):
    """Converts s-expression string into a list of significant tokens."""
    return s_expr.replace('(', ' ( ').replace(')', ' ) ').split()

def parse_sexpr(tokens):
    """
    Recursive parser that converts tokens into a TreeNode hierarchy.
    Handles standard (HEAD TAIL) and list-grouping ((...)) structures.
    """
    if len(tokens) == 0:
        raise ValueError("Unexpected end of input")
    
    token = tokens.pop(0)
    
    if token == '(':
        if tokens[0] == '(':
            #((NOT A) B) -> No explicit label, treated as implicit 'GROUP'
            node = TreeNode("GROUP") 
        else:
            node = TreeNode(tokens.pop(0))
        
        while tokens[0] != ')':
            node.add_child(parse_sexpr(tokens))
        
        tokens.pop(0)
        return node
    elif token == ')':
        raise ValueError("Unexpected )")
    else:
        # If the token is a leaf node/literal's liek A, B ...
        return TreeNode(token)


def add_arg(expr: str, new_arg: str) -> str:
    expr = expr.strip()
    if "$" in expr:
        return expr.replace("$", new_arg, 1)

    if not expr.endswith(")"):
        raise ValueError("Not a valid S‑expression")
    return expr[:-1] + " " + new_arg + ")"

def replace_one_symbol(expr: str, old: str, new: str) -> str:
    """
    Replace exactly one occurrence of `old` as a separate token with `new`,
    preserving the order of knobs in the expression.
    """
    # Use word boundaries to match whole tokens only
    pattern = r'\b' + re.escape(old) + r'\b'
    return re.sub(pattern, new, expr, count=1)

def exclude_one_symbol(expr:str , symbol_to_remove:str)->str:
    """
    Remove exactly one occurrence of `symbol_to_remove` as a separate token,
    preserving the order of knobs in the expression.
    """
    # Use word boundaries to match whole tokens only
    pattern = r'\b' + re.escape(symbol_to_remove) + r'\b'
    result = re.sub(pattern, '', expr, count=1)
    # Clean up multiple spaces and spaces before closing parentheses
    result = re.sub(r'\s+', ' ', result)  # Multiple spaces to single space
    result = re.sub(r'\s+\)', ')', result)  # Space before closing paren
    result = re.sub(r'\(\s+', '(', result)  # Space after opening paren
    return result.strip()

def isOP(token: str) -> bool:
    return token in LOGIC_OPERATORS


def _is_valid_logic_node(node: TreeNode, is_root: bool, allow_empty_and_root: bool) -> bool:
    if node.label == "NOT":
        return len(node.children) == 1 and all(
            _is_valid_logic_node(child, False, allow_empty_and_root)
            for child in node.children
        )

    if node.label == "OR":
        return len(node.children) >= 2 and all(
            _is_valid_logic_node(child, False, allow_empty_and_root)
            for child in node.children
        )

    if node.label == "AND":
        if len(node.children) == 0:
            return allow_empty_and_root and is_root
        return all(
            _is_valid_logic_node(child, False, allow_empty_and_root)
            for child in node.children
        )

    if isOP(node.label):
        return False

    return all(
        _is_valid_logic_node(child, False, allow_empty_and_root)
        for child in node.children
    )


def is_valid_logic_expr(expr: str, allow_empty_and_root: bool = True) -> bool:
    tokens = tokenize(expr)
    if not tokens:
        return False

    for index, token in enumerate(tokens):
        if isOP(token):
            prev_token = tokens[index - 1] if index > 0 else None
            if prev_token != '(':
                return False

    try:
        root = parse_sexpr(tokens[:])
    except Exception:
        return False

    return _is_valid_logic_node(root, is_root=True, allow_empty_and_root=allow_empty_and_root)


def prune_duplicate_children(expr_or_node: Union[str, TreeNode]) -> Union[str, TreeNode]:
    """
    Removes duplicate children from commutative AND/OR nodes.

    Accepts either:
    - a TreeNode (mutates in place and returns the same node), or
    - an s-expression string (returns the cleaned string).
    """
    def _prune_node(node: TreeNode) -> None:
        if not node.children:
            return

        for child in node.children:
            _prune_node(child)

        if node.label in ["AND", "OR"]:
            unique_children = []
            seen = set()
            for child in node.children:
                child_repr = str(child)
                if child_repr not in seen:
                    seen.add(child_repr)
                    unique_children.append(child)
            node.children = unique_children

    if isinstance(expr_or_node, TreeNode):
        _prune_node(expr_or_node)
        return expr_or_node

    if isinstance(expr_or_node, str):
        try:
            root = parse_sexpr(tokenize(expr_or_node))
            _prune_node(root)
            return str(root)
        except Exception:
            return expr_or_node

    return expr_or_node

def get_top_level_features(s_expr_str):
    """
    Parses "(AND A (OR B C) D)" into -> ["A", "(OR B C)", "D"]
    """
    content = s_expr_str.strip()
    if content.startswith("(AND"):
        content = content[4:-1].strip()
    elif content.startswith("(OR"):
        content = content[3:-1].strip()
    else:
        return s_expr_str

    
    features = []
    buffer = ""
    balance = 0
    
    for char in content:
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
            
        if char == ' ' and balance == 0:
            if buffer.strip():
                features.append(buffer.strip())
            buffer = ""
        else:
            buffer += char
            
    if buffer.strip():
        features.append(buffer.strip())
        
    return features

def isSymbol(s_expression):
    s_expression = s_expression.strip()
    # Single token -> symbol
    if "(" not in s_expression and ")" not in s_expression:
        return True
    # Must start with '(' to be a proper expression
    if not s_expression.startswith("("):
        return True
    # Treat (NOT ...) as a symbol
    if s_expression.startswith("(NOT"):
        return True
    # Nested only if it begins with (AND ...) or (OR...)
    return  not (s_expression.startswith("(AND") or s_expression.startswith("(OR"))