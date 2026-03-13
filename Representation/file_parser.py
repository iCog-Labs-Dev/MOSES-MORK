import csv
from pathlib import Path


def _parse_value(value):
    """Convert string to bool, int, float, or str. Returns None for empty values."""
    if value is None or value == '':
        return None
    
    v = value.strip().upper()
    
    if v in ('1', 'TRUE', 'T', 'YES'):
        return True
    elif v in ('0', 'FALSE', 'F', 'NO'):
        return False
    
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return value.strip()


def parse_file(filepath, delimiter=None, output_col='O'):
    """
    Parse CSV or table file with automatic delimiter detection.
    
    Args:
        filepath: Path to the file
        delimiter: Optional delimiter (auto-detected if None)
        output_col: Output column name to separate (default: 'O')
        
    Returns:
        (data_rows, output_values): Input features and output values
    """
    path = Path(filepath)
    
    if not path.exists():
        print(f"Error: File {filepath} not found.")
        return [], []
    
    if delimiter is None:
        delimiter = ',' if path.suffix.lower() == '.csv' else '\t'
    
    data_rows = []
    output_values = []
    
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                clean_row = {}
                
                for key, val in row.items():
                    parsed_val = _parse_value(val)
                    
                    if key == output_col:
                        output_values.append(parsed_val)
                    else:
                        clean_row[key] = parsed_val
                
                data_rows.append(clean_row)
                
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return [], []
    
    return data_rows, output_values
