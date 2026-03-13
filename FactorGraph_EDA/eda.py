"""
EDA (Estimation of Distribution Algorithm) for MOSES-MORK.

Bridges the DependencyMiner output to a PLN-weighted factor graph,
provides revision / deduction over the graph, and samples new
program instances from the learned distribution.
"""

import random
from copy import deepcopy
from typing import List, Optional, Tuple

from FactorGraph_EDA.pln import STV, c2w, w2c, revision, deduction, negation
from FactorGraph_EDA.factor_graph import SubtreeVariable, PairwiseFactor, FactorGraph
from Representation.representation import Instance, Knob, Deme, FitnessOracle
from Representation.selection import select_top_k
from Representation.helpers import get_top_level_features, isOP, tokenize

from DependencyMiner.miner import DependencyMiner


# ========================================================================
# 1.  Miner output  →  Factor graph structure
# ========================================================================

def build_factor_graph_from_miner(
    miner: DependencyMiner,
    dependencies: List[dict],
) -> FactorGraph:
    """
    Construct a FactorGraph from a fitted DependencyMiner.

    Variables: every unique subtree tracked in ``miner.single_weights``.
        marginal strength  = weight_x / total_weighted_contexts
        marginal confidence = w2c(single_count_x)   (more observations → more confident)

    Factors: each high-PMI pair from *dependencies*.
        stv = (strength, confidence)  already computed by the miner.
    """
    fg = FactorGraph()
    total = miner.total_weighted_contexts
    if total <= 0:
        return fg

    # --- variables (one per unique subtree) --------------------------------
    for name, weight in miner.single_weights.items():
        count = miner.single_counts.get(name, 0)
        s = weight / total                # marginal probability
        c = w2c(float(count))             # more observations → higher confidence
        fg.add_variable(SubtreeVariable(name, marginal_stv=(s, min(c, 0.9999))))

    # --- factors (one per meaningful dependency) ---------------------------
    for dep in dependencies:
        pair_str = dep["pair"]            # e.g. "A -- (NOT B)"
        parts = pair_str.split(" -- ", 1)
        if len(parts) != 2:
            continue
        var_a, var_b = parts[0].strip(), parts[1].strip()

        # Only create factor if both variables exist
        if fg.get_variable(var_a) is None or fg.get_variable(var_b) is None:
            continue

        stv: STV = (dep["strength"], dep["confidence"])
        fg.add_factor(PairwiseFactor(var_a, var_b, stv))

    return fg


# ========================================================================
# 2.  PLN revision across generations
# ========================================================================

def revise_factor_graph(new_fg: FactorGraph, old_fg: FactorGraph) -> None:
    """
    Merge *old_fg* evidence into *new_fg* in-place using PLN revision.

    For each variable / factor present in both graphs, their STVs are revised
    (evidence accumulates, confidence grows).  Items only in *new_fg* are kept
    as-is; items only in *old_fg* are carried forward with decayed confidence.
    """
    # --- revise shared variables -------------------------------------------
    for name, new_var in new_fg.variables.items():
        old_var = old_fg.get_variable(name)
        if old_var is not None:
            new_var.marginal_stv = revision(new_var.marginal_stv,
                                            old_var.marginal_stv)

    # carry forward old-only variables with slight confidence decay
    for name, old_var in old_fg.variables.items():
        if name not in new_fg.variables:
            decayed_c = old_var.marginal_stv[1] * 0.9
            new_fg.add_variable(
                SubtreeVariable(name, (old_var.marginal_stv[0], decayed_c)))

    # --- revise shared factors ---------------------------------------------
    for key, new_fac in new_fg.factors.items():
        old_fac = old_fg.get_factor(key)
        if old_fac is not None:
            new_fac.stv = revision(new_fac.stv, old_fac.stv)

    for key, old_fac in old_fg.factors.items():
        if key not in new_fg.factors:
            decayed_c = old_fac.stv[1] * 0.9
            new_fg.add_factor(
                PairwiseFactor(old_fac.var_a, old_fac.var_b,
                               (old_fac.stv[0], decayed_c)))


# ========================================================================
# 3.  PLN deduction — fill structural gaps
# ========================================================================

def apply_deduction(fg: FactorGraph) -> None:
    """
    For every pair of factors that share a variable (A-B and B-C),
    infer A-C via PLN deduction if no direct factor exists yet.

    Inferred factors are marked ``inferred=True`` and have lower
    confidence (reflecting that they are derived, not observed).
    """
    factor_list = list(fg.factors.values())
    new_factors: List[PairwiseFactor] = []

    for i in range(len(factor_list)):
        for j in range(i + 1, len(factor_list)):
            f1 = factor_list[i]
            f2 = factor_list[j]

            # find shared pivot variable
            vars1 = {f1.var_a, f1.var_b}
            vars2 = {f2.var_a, f2.var_b}
            shared = vars1 & vars2

            if not shared:
                continue

            pivot = shared.pop()
            remain_a = vars1 - {pivot}
            remain_c = vars2 - {pivot}

            # skip self-loops or degenerate factors
            if not remain_a or not remain_c:
                continue

            end_a = remain_a.pop()
            end_c = remain_c.pop()

            # skip if both endpoints are the same variable
            if end_a == end_c:
                continue

            # skip if direct factor already exists
            key = (min(end_a, end_c), max(end_a, end_c))
            if fg.get_factor(key) is not None:
                continue

            # marginal strength of the pivot
            pivot_var = fg.get_variable(pivot)
            s_b = pivot_var.marginal_stv[0] if pivot_var else 0.5

            stv_ac = deduction(f1.stv, f2.stv, s_b)
            new_factors.append(
                PairwiseFactor(end_a, end_c, stv_ac, inferred=True))

    for f in new_factors:
        fg.add_factor(f)


# ========================================================================
# 4.  Sampling from the factor graph
# ========================================================================

def _conditional_strength(
    factor: PairwiseFactor,
    assigned_var: str,
    assigned_present: bool,
) -> float:
    """
    Given a pairwise factor and that *assigned_var* is present / absent,
    return the conditional probability that the *other* variable is present.

    If assigned_present:
        P(other | assigned) ≈ factor.strength  (they co-occur)
    else:
        P(other | ¬assigned) ≈ 1 - factor.strength  (inverted)

    Confidence modulates: low confidence → fall back to 0.5 (uniform).
    """
    s, c = factor.stv
    if not assigned_present:
        s = 1.0 - s
    # blend toward 0.5 when confidence is low
    return c * s + (1.0 - c) * 0.5


def sample_from_factor_graph(
    fg: FactorGraph,
    n: int,
    root_op: str,
    all_knobs: List[Knob],
) -> List[Instance]:
    """
    Generate *n* new program instances by ancestral sampling from *fg*.

    Procedure for each sample:
      1. Order variables by marginal strength (strongest first).
      2. Walk through the ordered list; for each variable decide
         present / absent using marginal × conditional from already-
         assigned neighbours.
      3. Collect all "present" subtrees and form ``(root_op t1 t2 ...)``.
      4. Build an ``Instance`` with the matching knobs.

    Returns a list of *n* ``Instance`` objects (scores initialised to 0.0).
    """
    if not fg.variables:
        return []

    var_names = sorted(
        fg.variables.keys(),
        key=lambda v: fg.variables[v].marginal_stv[0],
        reverse=True,
    )

    knob_lookup = {k.symbol: k for k in all_knobs}
    instances: List[Instance] = []

    for idx in range(n):
        assigned: dict[str, bool] = {}
        present_subtrees: List[str] = []

        for vname in var_names:
            var = fg.variables[vname]
            s_marginal, c_marginal = var.marginal_stv

            # Start with marginal (blended toward 0.5 by confidence)
            p_present = c_marginal * s_marginal + (1.0 - c_marginal) * 0.5

            # Adjust with conditional information from assigned neighbours
            for fac in fg.neighbors(vname):
                other = fac.var_b if fac.var_a == vname else fac.var_a
                if other in assigned:
                    cond = _conditional_strength(fac, other, assigned[other])
                    # multiplicative blend
                    p_present *= cond
                    # re-normalise into [0.05, 0.95] to keep exploration
                    p_present = max(0.05, min(0.95, p_present))

            present = random.random() < p_present
            assigned[vname] = present
            if present:
                present_subtrees.append(vname)

        # --- reconstruct S-expression -------------------------------------
        if not present_subtrees:
            # fallback: pick at least one real knob symbol
            present_subtrees = [random.choice(sorted(knob_symbols))]

        if len(present_subtrees) == 1:
            expr = present_subtrees[0]
        else:
            inner = " ".join(present_subtrees)
            expr = f"({root_op} {inner})"

        # --- build knob list for the new instance --------------------------
        inst_knobs: List[Knob] = []
        seen_knob_symbols: set = set()
        for st in present_subtrees:
            # strip (NOT ...) wrappers to find base symbol
            base = st.strip()
            if base.startswith("(NOT ") and base.endswith(")"):
                base = base[5:-1].strip()
            # try base symbol first, then full subtree string
            for candidate in (base, st):
                if candidate in knob_lookup and candidate not in seen_knob_symbols:
                    inst_knobs.append(deepcopy(knob_lookup[candidate]))
                    seen_knob_symbols.add(candidate)

        instances.append(Instance(
            value=expr,
            id=idx + 1,
            score=0.0,
            knobs=inst_knobs,
        ))

    return instances


# ========================================================================
# 5.  Variation / Mutation
# ========================================================================

def mutate_instance(
    inst: Instance,
    all_knobs: List[Knob],
    mutation_rate: float = 0.3,
    fg: Optional[FactorGraph] = None,
) -> Instance:
    """
    Apply structural mutations to a sampled instance to introduce diversity.

    Mutations (applied probabilistically):
      1. **Negate** – wrap a random feature in ``(NOT ...)`` or unwrap an existing NOT.
      2. **Add feature** – include a knob variable not yet present.
      3. **Flip root operator** – switch ``AND`` ↔ ``OR``.
      4. **Create sub-expression** – wrap a pair of features in the alternate operator.

    The *mutation_rate* controls the probability of each individual mutation
    firing.  If a *fg* is supplied its marginal STVs guide which features
    are more likely to be negated (low-strength → more likely).
    """
    child = deepcopy(inst)
    features = get_top_level_features(child.value)

    # get_top_level_features returns a plain string for bare atoms
    if isinstance(features, str):
        features = [features] if features.strip() else []

    # --- detect root operator -----------------------------------------------
    val = child.value.strip()
    root_op = "AND"
    if val.startswith("(OR"):
        root_op = "OR"

    mutated_features = list(features)
    knob_symbols = {k.symbol for k in all_knobs}

    # symbols already present (base form, ignoring NOT wrappers)
    present_bases = set()
    for f in features:
        base = f.strip()
        if base.startswith("(NOT ") and base.endswith(")"):
            base = base[5:-1].strip()
        present_bases.add(base)

    # --- 1. Negate random features ------------------------------------------
    for i in range(len(mutated_features)):
        if random.random() >= mutation_rate:
            continue
        feat = mutated_features[i]
        # Use FG marginal to decide: low-strength features are more likely
        # to benefit from negation.
        negate_bias = 0.5
        if fg is not None:
            var = fg.get_variable(feat)
            if var is not None:
                negate_bias = 1.0 - var.marginal_stv[0]  # invert strength

        if feat.startswith("(NOT ") and feat.endswith(")"):
            mutated_features[i] = feat[5:-1].strip()        # unwrap
        elif random.random() < negate_bias:
            mutated_features[i] = f"(NOT {feat})"           # wrap

    # --- 2. Add a missing feature -------------------------------------------
    if random.random() < mutation_rate * 0.5:
        missing = [s for s in knob_symbols if s not in present_bases]
        if missing:
            new_feat = random.choice(missing)
            if random.random() < 0.3:
                new_feat = f"(NOT {new_feat})"
            mutated_features.append(new_feat)

    # --- 3. Flip root operator occasionally ---------------------------------
    if random.random() < mutation_rate * 0.15:
        root_op = "OR" if root_op == "AND" else "AND"

    # --- 4. Wrap a pair of features in a sub-expression ---------------------
    if len(mutated_features) >= 3 and random.random() < mutation_rate * 0.25:
        sub_op = "OR" if root_op == "AND" else "AND"
        idx1, idx2 = random.sample(range(len(mutated_features)), 2)
        f1, f2 = mutated_features[idx1], mutated_features[idx2]
        sub_expr = f"({sub_op} {f1} {f2})"
        for idx in sorted([idx1, idx2], reverse=True):
            mutated_features.pop(idx)
        mutated_features.append(sub_expr)

    # --- reconstruct S-expression -------------------------------------------
    if not mutated_features:
        mutated_features = [random.choice(list(knob_symbols))]

    if len(mutated_features) == 1:
        child.value = mutated_features[0]
    else:
        child.value = f"({root_op} {' '.join(mutated_features)})"

    # --- update knobs -------------------------------------------------------
    present_tokens = set(tokenize(child.value))
    knob_lookup = {k.symbol: k for k in all_knobs}
    child.knobs = [deepcopy(knob_lookup[s]) for s in present_tokens
                   if s in knob_lookup]
    child.score = 0.0
    child.id = random.randint(1000, 9999)
    return child


# ========================================================================
# 6.  EDA generation loop
# ========================================================================

def eda_generation(
    deme: Deme,
    fitness_oracle: FitnessOracle,
    top_k: int = 10,
    min_pmi: float = 0.1,
    min_freq: int = 2,
    sample_size: Optional[int] = None,
    prev_factor_graph: Optional[FactorGraph] = None,
    all_knobs: Optional[List[Knob]] = None,
) -> Tuple[Deme, FactorGraph]:
    """
    One generation of the EDA:

    1. Evaluate fitness of current instances.
    2. Select the top-k fittest.
    3. Mine sibling co-occurrence dependencies (weighted by fitness).
    4. Build a factor graph from the miner output.
    5. Optionally revise with the previous generation's factor graph (PLN).
    6. Apply deduction to fill structural gaps.
    7. Sample a new population from the factor graph.
    8. Keep elite instances alongside new samples.
    9. Return the updated deme and factor graph.
    """
    # -- 1. evaluate --------------------------------------------------------
    for inst in deme.instances:
        fitness_oracle.get_fitness(inst)

    # -- 2. select ----------------------------------------------------------
    k = min(top_k, len(deme.instances))
    top_instances = select_top_k(deme, k)

    if not top_instances:
        fg = FactorGraph()
        return deme, fg

    # -- 3. mine ------------------------------------------------------------
    expressions = [inst.value for inst in top_instances]
    weights = [inst.score for inst in top_instances]

    miner = DependencyMiner()
    miner.fit(expressions, weights)
    dependencies = miner.get_meaningful_dependencies(
        min_pmi=min_pmi, min_freq=min_freq)

    # -- 4. build factor graph ----------------------------------------------
    fg = build_factor_graph_from_miner(miner, dependencies)

    # -- 5. revise with previous generation ---------------------------------
    if prev_factor_graph is not None:
        revise_factor_graph(fg, prev_factor_graph)

    # -- 6. deduction -------------------------------------------------------
    apply_deduction(fg)

    # -- 7. sample ----------------------------------------------------------
    # detect root operator from the best instance
    best_expr = top_instances[0].value if top_instances else ""
    root_op = "AND"
    if isinstance(best_expr, str) and best_expr.startswith("("):
        tokens = best_expr.strip()[1:].split()
        if tokens and isOP(tokens[0]):
            root_op = tokens[0]

    # collect all knobs: prefer caller-supplied global knobs, else gather from population
    if all_knobs is None:
        all_knobs_local: List[Knob] = []
        seen_symbols: set = set()
        for inst in deme.instances:
            for kb in inst.knobs:
                if kb.symbol not in seen_symbols:
                    all_knobs_local.append(kb)
                    seen_symbols.add(kb.symbol)
    else:
        all_knobs_local = list(all_knobs)

    pop_size = sample_size or len(deme.instances)

    # If the factor graph has variables, sample from it; otherwise keep top instances
    if fg.variables:
        new_instances = sample_from_factor_graph(fg, pop_size, root_op, all_knobs_local)
    else:
        # No structure learned — duplicate top instances with slight variation
        new_instances = [deepcopy(inst) for inst in top_instances]

    # -- 7b. apply variation / mutation to introduce structural diversity ----
    mut_rate = deme.q_hyper.mutation_rate if hasattr(deme, 'q_hyper') else 0.3
    mutated: List[Instance] = []
    for inst in new_instances:
        if random.random() < mut_rate:
            mutated.append(mutate_instance(inst, all_knobs_local,
                                           mutation_rate=mut_rate, fg=fg))
        else:
            mutated.append(inst)
    new_instances = mutated

    # -- 8. evaluate new instances ------------------------------------------
    for inst in new_instances:
        fitness_oracle.get_fitness(inst)

    # -- 9. merge: keep elite + new, deduplicate, trim to pop_size ----------
    elite_count = max(1, k // 2)
    elites = top_instances[:elite_count]

    merged: dict[str, Instance] = {}
    for inst in elites:
        merged[inst.value] = inst
    for inst in new_instances:
        if inst.value not in merged or inst.score > merged[inst.value].score:
            merged[inst.value] = inst

    combined = sorted(merged.values(), key=lambda x: x.score, reverse=True)
    deme.instances = combined[:pop_size]

    deme.factor_graph = fg
    deme.generation += 1

    return deme, fg


def run_deme_eda(
    deme: Deme,
    fitness_oracle: FitnessOracle,
    num_generations: int = 20,
    top_k: int = 5,
    min_pmi: float = 0.0,
    min_freq: int = 1,
    sample_size: Optional[int] = None,
    all_knobs: Optional[List[Knob]] = None,
    verbose: bool = False,
) -> Tuple[Instance, FactorGraph]:
    """
    Run *num_generations* of EDA on a single deme.

    Returns the fittest Instance found across all generations and the
    final FactorGraph.
    """
    prev_fg: Optional[FactorGraph] = None
    best_ever: Optional[Instance] = None

    for gen in range(num_generations):
        deme, fg = eda_generation(
            deme,
            fitness_oracle,
            top_k=top_k,
            min_pmi=min_pmi,
            min_freq=min_freq,
            sample_size=sample_size,
            prev_factor_graph=prev_fg,
            all_knobs=all_knobs,
        )
        prev_fg = fg

        # track best instance across all generations
        gen_best = max(deme.instances, key=lambda x: x.score)
        if best_ever is None or gen_best.score > best_ever.score:
            best_ever = deepcopy(gen_best)

        if verbose:
            print(f"  Gen {gen+1:>3}/{num_generations}  "
                  f"pop={len(deme.instances):>3}  "
                  f"best={gen_best.score:.4f}  "
                  f"expr={gen_best.value}")

        # early stop if perfect score
        if gen_best.score >= 1.0:
            if verbose:
                print(f"  ** Perfect score reached at generation {gen+1} **")
            break

    return best_ever, fg

