from typing import List
import re

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