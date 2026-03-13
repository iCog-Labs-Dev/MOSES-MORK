# MOSES-MORK

![MOSES-MORK Architecture](Resources/MOSES-MORK.png)

---

## Table of Contents

1. [Installation & Running](#-installation--running)
2. [Core Concepts](#-core-concepts)
3. [Architecture](#-architecture)
4. [Methods & Algorithms](#-methods--algorithms)
5. [Repository Structure](#-repository-structure)
6. [Goals](#-goals)

---

**MOSES-MORK** is a Python implementation of a scalable, evolutionary program search workflow, designed to align with the **MORK / Hyperon Atomspace** ecosystem. It learns boolean programs (S-expressions) from truth-table data by iteratively improving them through sampling, selection, dependency mining, and probabilistic variation.

---

## 🚀 Installation & Running

### 1. Install Dependencies
From the repository root:
```bash
pip install -r requirements.txt
```
> **Key dependency:** `hyperon==0.2.9` (used for MeTTa integration and reducer invocation).

### 2. Run the Demo
```bash
python main.py
```

By default, the script:
- Loads `example_data/test_bin.csv` with output column `O`.
- Starts from the exemplar `(AND)`.
- Runs the workflow using the beta strategy (`fg_type="beta"`).

To switch to the alpha strategy, modify `main.py`:
```python
fg_type="alpha"
```

### 3. Run Tests
Execute the unit test runner starting from the project root:
```bash
python scripts/run_tests.py
```

---

## 🧠 Core Concepts

**MOSES** (Meta-Optimizing Semantic Evolutionary Search) is a framework combining semantic program representations, Estimation-of-Distribution Algorithms (EDA), and structure-preserving variation.

High-level workflow:
1. **Sampling:** Create candidate program "demes" (neighborhoods) around an exemplar.
2. **Selection:** Filter for the fittest candidates.
3. **Dependency Mining:** Discover correlated sub-expressions.
4. **Factor Graph Construction:** Build models from mined dependencies.
5. **Variation:** Apply crossover and mutation, guided by learned values.
6. **Canonicalization:** Reduce programs via Elegant Normal Form (ENF).

### Strategies
- **Alpha Path** (`run_abp_moses`): Deme sampling + standard EDA on each deme.
- **Beta Path** (`run_bp_moses`): Uses a Beta-distribution factor graph and evidence propagation to drive variation.

---

## 🏗 Architecture

### Program Representation
Programs are represented using **gCoDD** (grounded Combinatory Decision DAGs):
* **Internal nodes:** Combinatory structure.
* **Leaves:** Grounded predicates.
* **DAG structure:** Enables sharing, compression, and efficient rewrites.

### Canonicalization
To ensure semantic equivalence and reduce redundancy:
* **ENF (Elegant Normal Form):** Canonicalizes program graphs. ([Learn more](Resources/docs/ENF.md))
* **CENF (Correlation-Adapted ENF):** Preserves correlated structures discovered during learning.

### Variation Operators
Variation utilizes **quantale-based operations**:
* **Mask-based Crossover:** Operates on shared subgraphs.
* **Mutation:** Localized structural rewrites.
* All operators preserve semantic validity within the Atomspace. ([Learn more](Resources/docs/VariationalQuantale.md))

---

## 📊 Methods & Algorithms

### Sampling & Selection
* **Sampling:** Uses `randomBernoulli` and `randomUniform` sampling to explore demes around a central exemplar.
* **Selection:** Picks the best instances for subsequent crossover and mutation.
([Learn more](Resources/docs/sampling_and_selection.md))

### Estimation of Distribution (EDA)
EDA is performed via **quantale-valued factor graphs** embedded in Atomspace:
* Variables map to program components.
* Factors encode learned correlations.
* **Inference:** Belief propagation via message-passing rewrites.
* **Sampling:** Generates new candidates consistent with learned distributions.
([Learn more](Resources/docs/FactorGraph.md))

### Rewrites & Scalability
* Efficient local rewrites handle crossover, mutation, ENF/CENF, and pattern mining.
* **Complexity:** Proportional to the size of affected subgraphs (no global traversals).
* **Scalability:** Supports parallel execution and distributed Atomspace shards.

### System Flow & Diagrams
Visual representations of the system architecture, including the Alpha/Beta strategy paths and the ENF reduction pipeline, vary based on configuration.
([View System Flow Charts](Resources/docs/flow_charts.md))

---

## 📂 Repository Structure

| Directory / File | Description |
| :--- | :--- |
| `main.py` | Runnable script + unified `run_moses(...)` wrapper. |
| `Moses/` | Outer loop implementations. |
| `Representation/` | Core data structures, sampling, and selection utilities. |
| `DependencyMiner/` | Pattern/dependency mining over program trees. |
| `FactorGraph_EDA/` | Factor-graph EDA pipeline and Beta factor graph. |
| `Variation_quantale/` | Crossover and mutation operators. |
| `Feature_selection_algo/` | Feature selection used during deme sampling. |
| `reduct/` | ENF reducer and Hyperon atom registration. |
| `example_data/` | Example truth tables (CSV). |
| `scripts/` | Utility scripts (e.g., test runner). |
| `Resources/` | Diagrams and documentation. |

---

## 🎯 Goals

* Driving program evolution via algebraic logic and probabilistic reasoning
* Provide Closed-loop evolutionary learning system that demonstrates variation and belief propagation.
* Enable distributed, probabilistic program synthesis.
* Serve as a foundation for future AGI-oriented learning systems.