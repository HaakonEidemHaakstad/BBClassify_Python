import os
import sys
import bbclassify

def prep_data(filename):
    with open (filename, 'r') as file:
        lines = file.readlines()
    
    lines = [i.lower().split() for i in lines]
    lines = [[float(i) if i.replace(".", "", 1).replace("-", "", 1).isdigit() else i for i in j] for j in lines]
    return lines[0:3]



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
