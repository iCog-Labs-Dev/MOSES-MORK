# Results for test_bin.csv with output_col='O' and the following s_expr:
results = {
    "(AND E F (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625,
    "(AND E (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625,
    "(AND E (OR (AND C) (AND (NOT D))) (OR (AND F) (AND C)) (OR (AND (NOT C)) (AND D)))": 0.65625,
    "(AND E (OR (AND F) (AND C)) (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625,
    "(AND F (OR (AND (NOT C)) (AND D)) (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625,
    "(AND (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))) (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625,
    "(AND (OR (AND (NOT D)) (AND E)) (OR (AND F) (AND C)) (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625,
    "(AND E F (OR (AND (NOT C)) (AND C)) (OR (AND C D) (AND (NOT C) (NOT D))))": 0.65625,
    "(AND F (OR (AND (NOT C)) (AND E)) (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625,
    "(AND (OR (AND (NOT C)) (AND D)) (OR (AND C) (AND (NOT D))))": 0.65625
  }

