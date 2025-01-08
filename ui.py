import os
import sys
import pandas as pd
import bbclassify

def read_and_parse_input(filename: str) -> list:
    if not os.path.isabs(filename):
        filename = os.path.join(os.path.abspath(__file__), filename)
    with open (filename, 'r') as file:
        lines = file.readlines()
    lines = [i.lower().split() for i in lines]
    lines = [[float(i) if i.replace(".", "", 1).replace("-", "", 1).isdigit() else i for i in j] for j in lines]
    datafile: str = lines[1][0]
    if not os.path.isabs(datafile):
        datafile = os.path.join(os.path.abspath(__file__), datafile)
    with open (datafile, 'r') as file:
        datalines = file.readlines()
    if lines[1][1].lower() == "r":
        data = [i[0] for i in datalines]
    elif lines[1][1].lower() == "f":
        data = [[i[0] for _ in range(i[1])] for i in datalines]
        data = [i for j in data for i in j]
    if len(lines[0]) == 3:
        lines[0].append(0)
    return lines[0:3]

trying = os.path.abspath(__file__)[::-1]
trying[trying.index("/") - 1:][::-1]
print(read_and_parse_input(trying[trying.index("/"):][::-1] + "ui_test/test_input"))

#print(os.path.isabs("ui_test/test_input"))
#print(os.path.abspath(__file__))
#trying = os.path.abspath(__file__)[::-1]
#print(trying[trying.index("/"):][::-1])


#
#def prepare_data(data: str, control_card_input: list):
#    if os.path.isabs(data):
#        file_path = data
#    else:
#        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.getcwd()
#        file_path = os.path.join(base_dir, data)
#    if not os.path.isfile(data):
#        raise FileNotFoundError(f"The file '{data}' does not exist at '{file_path}'")
#    with open (data, 'r') as file:
#        lines = file.readlines()
#    if all(len(i) == 1 for i in lines):
#        lines = [lines[i][0] for i in lines]
#        if any(isinstance(i, str) for i in lines[0]):
#            raise TypeError("All values in the observed-score data file must be numeric.")
#    elif all(len(i) == 2 for i in lines):
#        None
#    else:
#        raise ImportError("Observed-score data file must have at most two columns. Either a raw-score distribution (one column), or a frequency-distribution (two columns)")
#    if control_card_input[0[0].lower()] == "hb":
#        if not all(i % 1 == 0 for i in lines[0]):
#            raise ValueError("All observed-score values must be integers under the Hanson and Brennan approach.")

#def expand_values(data):
#    """Expand values based on their counts.
#
#
#    Args:
#        data (list of tuple): A list of tuples, where each tuple contains a value
#                              and its count.


#    Returns:
#        list: A list of values expanded according to their counts.
#    """
#    expanded_list = []
#    for value, count in data:
#        expanded_list.extend([value] * count)
#    return expanded_list

#def main():
#    """Main function to process a two-column CSV-like file."""
#    if len(sys.argv) != 3:
#        print("Usage: script.py <input_file> <output_file>")
#        sys.exit(1)

#    input_file = sys.argv[1]
#    output_file = sys.argv[2]

#    data = read_csv_file(input_file)
#    if not data:
#        print(f"No valid data found in '{input_file}'.")
#        sys.exit(1)

#    expanded_list = expand_values(data)

#    try:
#        with open(output_file, 'w') as outfile:
#            outfile.write("\n".join(expanded_list))
#        print(f"Expanded list written to {output_file}")
#    except Exception as e:
#        print(f"An error occurred while writing to '{output_file}': {e}")
#        sys.exit(1)

#    "".endswith

#if __name__ == "__main__":
#    main()


