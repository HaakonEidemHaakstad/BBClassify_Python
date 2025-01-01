import os
import sys
import pandas as pd
import bbclassify

def read_input(filename: str) -> list:
    if os.path.isabs(filename):
        file_path = filename
    else:
        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
        file_path = os.path.join(base_dir, filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist at '{file_path}'")    
    with open (filename, 'r') as file:
        lines = file.readlines()
    lines = [i.lower().split() for i in lines]
    lines = [[float(i) if i.replace(".", "", 1).replace("-", "", 1).isdigit() else i for i in j] for j in lines]
    
    if len(lines[0]) == 3:
        lines[0].append(0)
    
    # Input validation.

    ## Line 1
    if len(lines[0]) not in [3, 4]:
        raise ImportError("The first line of the input file must have at least 3 and at most 4 entries.")
    if lines[0[0].lower()] not in ["hb", "ll"]:
        raise ValueError(f"The first value in the first row of the input file ({filename}) must be either 'hb' (specifying the Hanson and Brennan procedure) or 'll' (specifying the Livingston and Lewis procedure).")
    if not isinstance(lines[0][1], (float, int)):
        raise TypeError(f"The second value in the first row of the input file ({filename}) must be a numeric value between 0 and 1 (not inclusive), representing test-score reliability.")
    if lines[0][2] not in [2, 2.0, 4, 4.0]:
        raise TypeError(f"The third value in the first row of the input file ({filename}) must be either 2 (for the two-parameter beta true-score distribution) or 4 (the four-parameter beta distribution).") 
    if len(lines[0] == 4) and (not isinstance(lines[0][3], (float, int)) or lines[0][3] % 1 != 0):
        raise TypeError(f"The fourth value in the first row of the input file ({filename}) must be an integer.")
    
    ## Line 2
    if len(lines[1]) not in [2, 4, 5]:
        raise ValueError("The second row of the input file ({filename}) must contain 2 (if raw-score moments are supplied), 4 (if a raw-score- or frequency distribution is supplied). If the Livingston and Lewis procedure is specified in line 1, an optional fifth value can be specified to indicate the minimum possible score to attain on the test (default is 0).")
    if not isinstance(lines[1][1], str) or lines[1][1] not in ["R", "r", "F", "f", "M", "m"]:
        raise ValueError(f"The second value in the second row of the input file ({filename}) must be either R or r (indicating that the contents of the data-file ({lines[1][0]} represent final test-scores), F or f (specifying that the contents of {lines[1][1]} represents a frequency distribution, or M or m (specifying that the {lines[1][1]} contains raw-score moments).)")
    if lines[1][1] in ["M", "m"] and len(lines[1]) != 2:
        raise ValueError(f"If moment input is specified, the data-file ({filename}) and the input type must be specified in the second ")
    if lines[1][1] not in ["M", "m"] and (not isinstance(lines[1][2], (float, int)) or lines[1][2] % 1 != 0):
        raise ValueError(f"If the second value in the second row of the input file ({filename}) is not M or m, the column of the data-file ({lines[1][1]}) that contains the test-scores must be specified as the third value in the second row of the input file ({filename}).") 
    if lines[1][1] in ["R", "r", "F", "f"] and (not isinstance(lines[1][3], (float, int)) or lines[1][3] % 1 != 0):
        raise ValueError(f"If the second value in the second row of the input file ({filename} is R or r, or F or f, the fourth value of the second row of the input file must be an integer specifying either (if a raw-score distribution is specified) the number of test items (for the Hanson and Brennan procedure) or the maximum possible test-score (for the Livingston and Lewis procedure), or (if a frequency distribution is specified) the column in the data-file ({filename}) containing the frequencies of sum-score occurances (if F or f).)")
    
    
    return lines[0:3]

def prepare_data(data: str, control_card_input: list):
    if os.path.isabs(data):
        file_path = data
    else:
        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
        file_path = os.path.join(base_dir, data)
    if not os.path.isfile(data):
        raise FileNotFoundError(f"The file '{data}' does not exist at '{file_path}'")
    with open (data, 'r') as file:
        lines = file.readlines()
    if all(len(i) == 1 for i in lines):
        lines = [lines[i][0] for i in lines]
        if any(isinstance(i, str) for i in lines[0]):
            raise TypeError("All values in the observed-score data file must be numeric.")
    elif all(len(i) == 2 for i in lines):
        None
    else:
        raise ImportError("Observed-score data file must have at most two columns. Either a raw-score distribution (one column), or a frequency-distribution (two columns)")
    if control_card_input[0[0].lower()] == "hb":
        if not all(i % 1 == 0 for i in lines[0]):
            raise ValueError("All observed-score values must be integers under the Hanson and Brennan approach.")

def expand_values(data):
    """Expand values based on their counts.

    Args:
        data (list of tuple): A list of tuples, where each tuple contains a value
                              and its count.

    Returns:
        list: A list of values expanded according to their counts.
    """
    expanded_list = []
    for value, count in data:
        expanded_list.extend([value] * count)
    return expanded_list

def main():
    """Main function to process a two-column CSV-like file."""
    if len(sys.argv) != 3:
        print("Usage: script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = read_csv_file(input_file)
    if not data:
        print(f"No valid data found in '{input_file}'.")
        sys.exit(1)

    expanded_list = expand_values(data)

    try:
        with open(output_file, 'w') as outfile:
            outfile.write("\n".join(expanded_list))
        print(f"Expanded list written to {output_file}")
    except Exception as e:
        print(f"An error occurred while writing to '{output_file}': {e}")
        sys.exit(1)

    "".endswith

if __name__ == "__main__":
    main()
