import hyperon

from reduct.enf.main import reduce
from .helpers import *

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Callable
from copy import deepcopy
import random

class Factor:
    """
    A simple factor that connects a set of variables (Instances) and
    optionally holds a potential function over them.
    """
    def __init__(
        self,
        variables: List["Instance"],
        potential: Callable[[List["Instance"]], float] | None = None,
        name: str | None = None,
    ) -> None:
        self.variables = variables
        self.potential = potential
        self.name = name or "factor"

    def evaluate(self) -> float:
        if self.potential is None:
            return 1.0
        return self.potential(self.variables)


class FactorGraph:
    def __init__(self, variables: List[Any], factors: List[Factor]) -> None:
        self.variables = variables
        self.factors = factors

    def add_variable(self, var: Any) -> None:
        self.variables.append(var)

    def add_factor(self, factor: Factor) -> None:
        self.factors.append(factor)

    def neighbors(self, var: Any) -> List[Factor]:
        """Return all factors that involve the given variable."""
        return [f for f in self.factors if var in f.variables]


class Quantale(ABC):
    @abstractmethod
    def join(self, parent1, parent2):
        """Join operation for the quantale."""
        pass

    @abstractmethod
    def product(self, parent1, parent2):
        """Product operation for the quantale."""
        pass
    
    @abstractmethod
    def residium(self, parent1, parent2):
        """Residium operation for the quantale."""
        pass

    @abstractmethod
    def unit(self):
        """Unit operation for the quantale."""
        pass

@dataclass
class Knob:
    symbol: str
    id: int
    Value: List[bool]

@dataclass
class Instance:
    value: Any
    id: int
    score: float
    knobs: List[Knob]

@dataclass
class Hyperparams:
    mutation_rate: float
    crossover_rate: float

class Deme(Quantale):
    def __init__(self, instances: List[Instance], id: str, q_hyper: Hyperparams) -> None:
        super().__init__()
        self.instances = instances
        self.id = id
        self.q_hyper = q_hyper
        self.generation = 0
        self.factor_graph : FactorGraph | None = None

    # TODO: implement variational quantale operations
    def join(self, parent1: Instance, parent2: Instance) -> Instance:
        pass

    def product(self, parent1: Instance, parent2: Instance) -> Instance:
        pass

    def residium(self, parent1: Instance, parent2: Instance) -> Instance:
        pass

    def unit(self) -> Instance:
        pass

    def construct(self) -> List[Instance]:
        if self.generation == 0:
            for instances in self.instances:
                expr = instances.value
                if "$" in expr:
                    for knob in instances.knobs:
                        expr = add_arg(expr, knob.symbol)
                        instances.value = expr
            
            if len(self.instances) == 1:
                parent = deepcopy(self.instances[0])
                new_instance = sample_random_instances(parent, self.q_hyper)
                self.instances.append(new_instance)
            
            self.factor_graph = build_factor_graph_from_deme(self)
        # TODO: have pattern miner to extract new dependencies between instances and update the factor graph
        #  also include pln for actions like probabilistic inference over the factor graph.
        return self.instances
    
    def to_tree(self):
        header = (
            "{Deme "
            + str(self.id)
            + " Generation: "
            + str(self.generation)
            + " Instances: "
            + str(len(self.instances))
            + "}"
        )

        # one entry per instance
        body = [
            {
                "id": instance.id,
                "expression": instance.value,
                "knobs": instance.knobs,
            }
            for instance in self.instances
        ]
        return [header, body]
        # return ["{Deme " + str(self.id) + " Generation: " + str(self.generation) + " Instances: " + str(len(self.instances)) + "}",
        #         [{"Expresstion": (instance.value, instance.knobs) for instance in self.instances}]]
    
def initialize_deme(program_sketch: str, ITable: List[dict]) -> Deme:
    """
    initialize_deme: Given an initial program sketch it constructs it with the available
        knobs and intitialize the deme.
    Args:
        program_sketch - An initial program sketch. Eg: (AND $ $), '$' represent
            positions of knobs.
    Returns: A deme object
    """
    knobs = knobs_from_truth_table(ITable)
    instances = [Instance(value=program_sketch, id=1, score=0.0, knobs=knobs)]
    hyperparams = Hyperparams(mutation_rate=0.1, crossover_rate=0.6)
    deme = Deme(instances=instances, id="deme-00", q_hyper=hyperparams)
    deme.construct()
    return deme

def sample_random_instances(instance: Instance, hyperparams: Hyperparams) -> Instance:
    """
    sample_random_instances: Given an instance/sketch it creates a child 
       by mutating the given instance parent. mutation is based on the 
       given mutation rate.
    Args:
        instance (Instance): The parent instance to mutate.
        hyperparams (Hyperparams): Hyperparameters including mutation rate.
    Returns: The new child instance.
    """
    child = deepcopy(instance)
    child.id += 1
    for knob in child.knobs:
        # new_value = []
        if random.random() < hyperparams.mutation_rate:
            old_symbol = deepcopy(knob.symbol)
            knob.symbol = f"(NOT {knob.symbol})"
            child.value = replace_one_symbol(child.value, old_symbol, knob.symbol)
    return child

def select_top_k(deme: Deme, k: int) -> List[Instance]:
    """
    select_top_k: Selects the top k instances from the deme based on their scores.
    Args:
        deme (Deme): The deme containing instances.
        k (int): The number of top instances to select.
    Returns: A list of the top k instances.
    """
    sorted_instances = sorted(deme.instances, key=lambda inst: inst.score, reverse=True)
    return sorted_instances[:k]

def knobs_from_truth_table(ITable: List[dict]) -> List[Knob]:
    """
    Given a truth table (list of dict rows), extract:
      - unique symbols (keys)
      - unique values per symbol
    and instantiate Knob objects.
    """
    if not ITable:
        return []

    values_by_key: dict[str, list[bool]] = {}

    for row in ITable:
        for key, val in row.items():
            if key not in values_by_key:
                values_by_key[key] = []
            if val not in values_by_key[key]:
                values_by_key[key].append(val)

    knobs: List[Knob] = []
    for idx, (symbol, vals) in enumerate(values_by_key.items(), start=1):
        knobs.append(Knob(symbol=symbol, id=idx, Value=vals))

    return knobs

def build_factor_graph_from_deme(deme: Deme) -> FactorGraph:
    """
    Build a factor graph where:
      - each Instance in the deme is a variable node
      - factors capture dependencies between related instances

    For now, we:
      * connect instances that share the same base expression skeleton
        (ignoring (NOT ...) wrappers) and / or
      * connect parent–child pairs by edit distance / knob mutations.
    """
    variables: List[Instance] = list(deme.instances)
    factors: List[Factor] = []

    # simple helper: strip NOT wrappers from a symbol
    def strip_not(symbol: str) -> str:
        symbol = symbol.strip()
        if symbol.startswith("(NOT ") and symbol.endswith(")"):
            return symbol[5:-1].strip()
        return symbol

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            inst_i = variables[i]
            inst_j = variables[j]

            if (isinstance(inst_i.value, str) and isinstance(inst_j.value, str)
                  and inst_i.value.split()[0] == inst_j.value.split()[0]):
                
                def potential(pair: List[Instance], _i=inst_i, _j=inst_j) -> float:
                    # example: higher potential if they share more base knobs
                    knobs_i = {strip_not(k.symbol) for k in _i.knobs}
                    knobs_j = {strip_not(k.symbol) for k in _j.knobs}
                    common = knobs_i & knobs_j
                    return 1.0 + len(common)

                factors.append(
                    Factor(
                        variables=[inst_i, inst_j],
                        potential=potential,
                        name=f"shared-structure: {inst_i.id}-{inst_j.id}",
                    )
                )

    return FactorGraph(variables=variables, factors=factors)
# -> program sketch of this structrure is possible: 
# (AND $ $)
# -> their can be any number of $/knobs in the sketch.
# -> Given a truth table some thing like the following instances we can define the knobs:
# [ {"A": True, "B": True, "O": True}, 
#   {"A": True, "B": False, "O": False},
#   {"A": False, "B": True, "O": False},
#   {"A": False, "B": False, "O": False}]
# 
# -> The knobs would be:
# [Knob("A", [True, False], Knob("B", [True, False])]; the result would be ["O", [True, False]]
# 
# -> A deme will look like this, at the start of the program:
# Deme(["(AND A B)"], "01", Hyperparams(0.1, 0.6))
# 
# -> An instance would be:
# Instance(value="(AND A B)", id=1, score=0, knobs=[Knob("A", [True, False]), Knob("B", [True, False]), Knob("O", [True, False])])
# 
# -> The deme would then contain a list of instances after the first mutation/crossover operations:
# -> But first before that we need to sample random instances to populate the deme initially.
# -> thus without changing the parent operator we will have random sampling of knobs from our truth table.
# def sample_random_instances(instance: Instance) -> Instance:
#     from the given instance/sketch randomly assign knob values
#     return new_instance
# -> Then we can populate the deme with a list of such instances.