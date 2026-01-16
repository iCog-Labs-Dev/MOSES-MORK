import csv
import io
from .representation import knobs_from_truth_table 


def load_truth_table(filepath, output_col='O'):
    """
    Reads a CSV file where the first row is headers (A, B, C...)
    and subsequent rows are truth values (0/1, T/F, True/False).
    
    Returns: A list of dictionaries, e.g., [{'A': True, 'B': False}, ...]
      and a list of target values for the output column e.g. [True, False, ...].
    """
    data_rows = []
    output_values = []    
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                clean_row = {}
                for key, val in row.items():
                    v = val.strip().upper()
                    bool_val = v in ('1', 'TRUE', 'T', 'YES')

                    if key == output_col:
                        output_values.append(bool_val)
                    else:
                        clean_row[key] = bool_val

                data_rows.append(clean_row)
                
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], []

    return data_rows, output_values



# data_rows, targets = load_truth_table('Representation/test.csv', output_col='O')
# print(knobs_from_truth_table(data_rows))

