# MOSES-MORK
![MOSES-MORK Architecture](Resources/MOSES-MORK.png)

## Overview

This repository outlines a scalable implementation of the **MOSES**
(**Meta-Optimizing Semantic Evolutionary Search**) framework within the **MORK**
implementation of the **Hyperon Atomspace**.

The design integrates evolutionary program search, probabilistic modeling,
and symbolic graph rewriting into a unified, distributed system. The core
idea is to represent programs, variation operators, and learned
distributions directly within Atomspace, enabling efficient local updates
and large-scale parallel execution.

---

## Core Concepts

### MOSES

MOSES is an evolutionary program learning framework that combines:

* Semantic program representations
* Estimation-of-Distribution Algorithms (EDA)
* Structure-preserving variation operators

### Hyperon Atomspace (MORK)

Atomspace provides a graph-based knowledge representation and rewrite engine.
MORK is a performant implementation suitable for distributed and parallel
execution.

---

## Architecture

### Program Representation

Programs are represented using **gCoDD**
(**grounded Combinatory Decision DAGs**):

* Internal nodes represent combinatory structure
* Leaves are grounded predicates
* DAG structure enables sharing, compression, and efficient rewrites

### Canonicalization

To ensure semantic equivalence and reduce search redundancy:

* **Elegant Normal Form (ENF)** is applied to canonicalize program graphs
* **Correlation-Adapted ENF (CENF)** extends ENF by preserving correlated
  structures discovered during learning

These normalization passes are implemented as local graph rewrites.

---

## Variation Operators

Variation is expressed using **quantale-based operations**:

* **Mask-based crossover** operates on shared subgraphs
* **Mutation** is implemented as localized structural rewrites
* All operators preserve semantic validity

Quantale semantics allow variation to be composed, analyzed, and executed
uniformly within Atomspace.

---

## Estimation of Distribution (EDA)

EDA is performed by embedding **quantale-valued factor graphs directly into
Atomspace**:

* Variables correspond to program components or decisions
* Factors encode learned dependencies and correlations
* **n-ary factor discovery** is driven by pattern mining over Atomspace graphs

### Inference and Sampling

* **Belief propagation** is implemented via Atomspace message-passing rewrites
* Sampling produces new candidate programs consistent with learned
  distributions

---

## Algorithms and Rewrites

Efficient local rewrites are provided for:

* Crossover and mutation
* ENF and CENF normalization
* Pattern mining for factor discovery
* Belief-propagation inference and sampling

Each operation only touches the relevant subgraphs, ensuring scalability.

---

## Complexity and Scalability

* Each MOSES step runs in time proportional to the size of affected subgraphs
  or variables
* No global graph traversals are required
* Naturally supports:

  * Parallel execution
  * Distributed Atomspace shards
  * Incremental updates

This makes the approach suitable for large-scale evolutionary search.

---

## Goals of This Repository

* Demonstrate a scalable MOSES implementation in Hyperon Atomspace
* Provide reusable Atomspace rewrite rules for evolutionary search
* Enable distributed, probabilistic program synthesis
* Serve as a foundation for future AGI-oriented learning systems

