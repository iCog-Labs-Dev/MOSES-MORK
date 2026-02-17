"""
Factor graph representation for the EDA approach.

Variables are subtrees (binary: present/absent in a program).
Factors are pairwise edges carrying PLN truth values (strength, confidence).
"""

from typing import Dict, List, Optional, Tuple
from Representation.pln import STV


class SubtreeVariable:
    """
    A variable node representing a subtree that can be present or absent
    in a candidate program.

    Attributes:
        name:         canonical string, e.g. "A", "(NOT B)", "(OR C D)"
        marginal_stv: (strength, confidence) — how likely this subtree is to
                      appear in a fit individual. strength ≈ P(present).
    """

    def __init__(self, name: str, marginal_stv: STV = (0.5, 0.0)):
        self.name = name
        self.marginal_stv = marginal_stv

    def __repr__(self):
        s, c = self.marginal_stv
        return f"Var({self.name}, stv=({s:.3f},{c:.3f}))"

    def __eq__(self, other):
        if not isinstance(other, SubtreeVariable):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class PairwiseFactor:
    """
    A factor (edge) connecting two subtree variables.

    The STV encodes: "given that var_a is present, how strongly does that
    predict var_b is also present?"  Higher strength + higher confidence
    means the two subtrees are tightly coupled.

    Attributes:
        var_a:    name of the first variable
        var_b:    name of the second variable
        stv:      (strength, confidence) — from the miner or PLN inference
        inferred: True if this factor was deduced (not directly observed)
    """

    def __init__(self, var_a: str, var_b: str, stv: STV,
                 inferred: bool = False):
        # Store in sorted order for consistent lookup
        if var_a > var_b:
            var_a, var_b = var_b, var_a
        self.var_a = var_a
        self.var_b = var_b
        self.stv = stv
        self.inferred = inferred

    @property
    def key(self) -> Tuple[str, str]:
        return (self.var_a, self.var_b)

    def __repr__(self):
        s, c = self.stv
        tag = " [deduced]" if self.inferred else ""
        return f"Factor({self.var_a} -- {self.var_b}, stv=({s:.3f},{c:.3f}){tag})"

    def __eq__(self, other):
        if not isinstance(other, PairwiseFactor):
            return False
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)


class FactorGraph:
    """
    A factor graph whose variable nodes are subtrees and whose factor
    (edge) nodes are pairwise couplings with PLN truth values.

    """

    def __init__(self):
        self.variables: Dict[str, SubtreeVariable] = {}
        self.factors: Dict[Tuple[str, str], PairwiseFactor] = {}

    # -- construction helpers ------------------------------------------------

    def add_variable(self, var: SubtreeVariable) -> None:
        self.variables[var.name] = var

    def add_factor(self, factor: PairwiseFactor) -> None:
        self.factors[factor.key] = factor

    def get_variable(self, name: str) -> Optional[SubtreeVariable]:
        return self.variables.get(name)

    def get_factor(self, key: Tuple[str, str]) -> Optional[PairwiseFactor]:
        # Normalise order
        a, b = key
        if a > b:
            a, b = b, a
        return self.factors.get((a, b))

    # -- neighbourhood queries -----------------------------------------------

    def neighbors(self, var_name: str) -> List[PairwiseFactor]:
        """Return all factors that involve the given variable."""
        return [f for f in self.factors.values()
                if f.var_a == var_name or f.var_b == var_name]

    def neighbor_names(self, var_name: str) -> List[str]:
        """Return names of all variables connected to *var_name*."""
        names = []
        for f in self.neighbors(var_name):
            other = f.var_b if f.var_a == var_name else f.var_a
            names.append(other)
        return names

    # -- summary -------------------------------------------------------------

    def __repr__(self):
        return (f"FactorGraph(vars={len(self.variables)}, "
                f"factors={len(self.factors)})")

