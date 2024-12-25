import sys
import bbclassify
import sys

def read_first_three_lines(file_path):
    """Read the first three lines of a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list of str: A list containing the first three lines of the file.
    """
    lines = []
    try:
        with open(file_path, 'r') as file:
            for _ in range(3):
                line = file.readline()
                if line == "":
                    break
                lines.append(line.strip())
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return lines

def process_content(content):
    """Process the content.

    Args:
        content (list of str): Lines of content to process.

    Returns:
        str: The processed content as a single string.
    """
    return "\n".join(line.upper() for line in content)

def main():
    """Main function to handle file input and output."""
    if len(sys.argv) != 3:
        print("Usage: script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    lines = read_first_three_lines(input_file)
    if not lines:
        print(f"No content to process in '{input_file}'.")
        sys.exit(1)

    processed_content = process_content(lines)

    try:
        with open(output_file, 'w') as outfile:
            outfile.write(processed_content)
        print(f"Processed content written to {output_file}")
    except Exception as e:
        print(f"An error occurred while writing to '{output_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
