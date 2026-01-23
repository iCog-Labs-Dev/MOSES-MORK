from typing import List
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
    tokens = re.findall(r'\(|\)|[^\s()]+', expr)
    for i, t in enumerate(tokens):
        if t == old:
            tokens[i] = new
            break
    out = []
    for t in tokens:
        if t == '(' or t == ')':
            out.append(t)
        else:
            if out and out[-1] != '(':
                out.append(' ')
            out.append(t)
    return ''.join(out)

def exclude_one_symbol(expr:str , symbol_to_remove:str)->str:
    """
    Remove exactly one occurrence of `symbol_to_remove` as a separate token,
    preserving the order of knobs in the expression.
    """
    tokens = re.findall(r'\(|\)|[^\s()]+', expr)
    for i, t in enumerate(tokens):
        if t == symbol_to_remove:
            del tokens[i]
            break
    out = []
    for t in tokens:
        if t == '(' or t == ')':
            out.append(t)
        else:
            if out and out[-1] != '(':
                out.append(' ')
            out.append(t)
    return ''.join(out)

def isOP(token: str) -> bool:
    return token in ['AND', 'OR', 'NOT']