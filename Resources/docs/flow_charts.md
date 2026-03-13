## 1) End-to-end flow charts

### 1.1 Overall system flow (main.py → strategy)
```mermaid
flowchart TD
  A["Start: main.py"] --> B["Load CSV truth table"]
  B --> C["Build knobs list"]
  C --> D["Create exemplar Instance (e.g., AND)"]
  D --> E["FitnessOracle scores exemplar"]
  E --> F{"fg_type?"}

  F -->|alpha| G["run_abp_moses"]
  F -->|beta| H["run_bp_moses"]
  F -->|other| G

  G --> I["Finalize metapop"]
  H --> I["Finalize metapop"]
  I --> J["Print top instances"]
  J --> K["End"]
```

### 1.2 Alpha path (ABP MOSES + EDA per deme)
```mermaid
flowchart TD
  A["run_abp_moses entry"] --> B{"max_iter <= 0"}
  B -->|yes| Z["return metapop"]

  B -->|no| C["sample demes from truth table"]
  C --> D["for each deme: run_deme_eda"]
  D --> E["collect best instance per deme"]
  E --> F["merge into metapop (dedupe by value)"]
  F --> G["pick new exemplar (stagnation handling)"]
  G --> H["recurse with max_iter minus 1"]
```

### 1.3 Beta path (BP-guided variation inside deme)
```mermaid
flowchart TD
  A["variation step"] --> B["select top k exemplars"]
  B --> C["fit DependencyMiner on values and weights"]
  C --> D["get meaningful dependencies"]
  D --> E["update BetaFactorGraph rules"]
  E --> F["set prior from top rule"]
  F --> G["run evidence propagation"]
  G --> H["build stv dictionary from node beliefs"]

  H --> I{"crossover possible"}
  I -->|yes| J["run crossTopOne and create children"]
  I -->|no| K["skip crossover"]

  J --> L["run mutation (additive and multiplicative)"]
  K --> L

  L --> M["reduce and score candidates"]
  M --> N["add unique candidates to deme"]
  N --> O["next generation or return"]
```

### 1.4 ENF reduction pipeline (reduct/enf)
```mermaid
flowchart TD
  A["reduce(metta, expr)"] --> B["Parse expression"]
  B --> C["Build binary expression tree"]
  C --> D["propagateTruthValue -> constraint tree"]
  D --> E["gatherJunctors normalize AND/OR"]
  E --> F["reduceToElegance apply reductions"]
  F --> G["constraint_tree_to_metta_expr"]
  G --> H["Return reduced expression"]
```
---

