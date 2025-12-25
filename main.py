from Representation.exp import *

def main(): 
    program_sketch = "(AND $ $)"
    ITable = [
    {"A": True,  "B": True,  "O": True},
    {"A": True,  "B": False, "O": False},
    {"A": False, "B": True,  "O": False},
    {"A": False, "B": False, "O": False},
    ]

    deme = initialize_deme(program_sketch, ITable)
    
    print(deme.to_tree())
    print("")

    instances = select_top_k(deme, k=2)
    print(f"selected instances: {instances}")
    print("")
    fg = deme.factor_graph
    for f in fg.factors:
        print(f.name, "->", [v.id for v in f.variables])
    
if __name__ == "__main__":
    main()