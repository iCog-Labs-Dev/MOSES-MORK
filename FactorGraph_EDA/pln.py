"""
PLN (Probabilistic Logic Networks) truth-value functions in Python.

Each truth value is a tuple (strength, confidence) abbreviated as STV.
  - strength ∈ [0, 1]: how likely/true the statement is
  - confidence ∈ [0, 1): how much evidence backs the estimate

Translated from pln.metta into Python for use in the EDA factor graph.
"""

from typing import Tuple

STV = Tuple[float, float]

# ---------------------------------------------------------------------------
# Core conversions  
# ---------------------------------------------------------------------------

def c2w(c: float) -> float:
    """Confidence → weight of evidence.  c2w(c) = c / (1 - c)."""
    if c >= 1.0:
        return float('inf')
    return c / (1.0 - c)


def w2c(w: float) -> float:
    """Weight of evidence → confidence.  w2c(w) = w / (w + 1)."""
    if w == float('inf'):
        return 1.0
    return w / (w + 1.0)


# ---------------------------------------------------------------------------
# Revision  
# ---------------------------------------------------------------------------

def revision(stv1: STV, stv2: STV) -> STV:
    """
    Merge two independent pieces of evidence about the same proposition.

    Uses weight-of-evidence merging:
        w_total = w1 + w2
        strength = (w1*s1 + w2*s2) / w_total
        confidence = w2c(w_total)

    Confidence is guaranteed to grow (or stay) — never shrink.
    """
    s1, c1 = stv1
    s2, c2 = stv2

    w1 = c2w(c1)
    w2 = c2w(c2)
    w = w1 + w2

    if w == 0:
        return (0.0, 0.0)

    s = (w1 * s1 + w2 * s2) / w
    c = w2c(w)

    return (min(1.0, s), min(0.9999, max(c, c1, c2)))


# ---------------------------------------------------------------------------
# Deduction  
# ---------------------------------------------------------------------------

def deduction(stv_ab: STV, stv_bc: STV, s_b: float) -> STV:
    """
    Transitive inference: if A→B with stv_ab and B→C with stv_bc,
    infer A→C.

    s_b is the marginal strength (base-rate) of B.

    Strength formula:
        s_ac = s_ab * s_bc  +  ((1 - s_ab) * (s_bc - s_b * s_bc)) / (1 - s_b)
    Confidence:
        c_ac = min(c_ab, c_bc)  — the chain is only as strong as its weakest link.
    """
    s_ab, c_ab = stv_ab
    s_bc, c_bc = stv_bc

    if s_b > 0.9999:
        s_ac = s_bc
    else:
        s_ac = s_ab * s_bc + ((1.0 - s_ab) * (s_bc - s_b * s_bc)) / (1.0 - s_b)

    s_ac = max(0.0, min(1.0, s_ac))
    c_ac = min(c_ab, c_bc)

    return (s_ac, w2c(c_ac))


# ---------------------------------------------------------------------------
# Inversion  (pln.metta Truth_inversion)
# ---------------------------------------------------------------------------

def inversion(stv_ab: STV, s_b: float, c_b: float) -> STV:
    """
    Given P(A|B) as stv_ab and marginal (s_b, c_b) for B,
    derive a (weaker) estimate of P(B|A).

    From pln.metta:
        strength = s_ab  (same direction strength, symmetric approximation)
        confidence = c_b * c_ab * 0.6   (penalised)
    """
    s_ab, c_ab = stv_ab
    c_ba = c_b * c_ab * 0.6
    return (s_ab, min(0.9999, c_ba))


# ---------------------------------------------------------------------------
# Negation  (pln.metta Truth_Negation)
# ---------------------------------------------------------------------------

def negation(stv: STV) -> STV:
    """Negate a truth value: strength flips, confidence stays."""
    s, c = stv
    return (1.0 - s, c)


# ---------------------------------------------------------------------------
# Modus Ponens  (PLN Book §5.7.1)
# ---------------------------------------------------------------------------

def modus_ponens(stv_a: STV, stv_ab: STV) -> STV:
    """
    Given A with stv_a and A→B with stv_ab, infer B.

    Strength: s_a * s_ab + 0.02 * (1 - s_a)
    Confidence: c_a * c_ab
    """
    s_a, c_a = stv_a
    s_ab, c_ab = stv_ab
    s_b = s_a * s_ab + 0.02 * (1.0 - s_a)
    c_b = c_a * c_ab
    return (max(0.0, min(1.0, s_b)), min(0.9999, c_b))

