import os
import sys
import numpy as np
import bbclassify
import statistics as stats
import scipy.stats

def read_and_parse_input(filename: str) -> list:
    input_error: str = "Input error. Execution terminated."
    if not os.path.isabs(filename):
        filename = os.path.join(os.path.dirname(sys.executable), filename)
    try:
        with open(filename, "r") as file:
            lines: list = file.readlines()
    except:
        raise ImportError("Error reading input file. Check whether the file-path is correctly specified.")
    lines: list = [i.lower().split() for i in lines]
    lines: list = [[float(i) if i.replace(".", "", 1).replace("-", "", 1).isdigit() else i for i in j] for j in lines]

    # Input validation.
    # Line 1.
    if lines[0][0].lower() not in ["ll", "hb"]:
        print(f"Procedure specification must be either LL for the Livingston and Lewis procedure, or HB for the Hanson and Brennan procedure. Current input is '{lines[0][0]}'.")
        raise TypeError(input_error)
    
    if not isinstance(lines[0][1], (float, int)):
        print(f"Reliability specification must be a numeric value between 0 and 1. Current input is '{lines[0][1]}'.")
        raise TypeError(input_error)
    
    if lines[0][0].lower() == "ll" and 0 >= lines[0][1] >= 1:
        print(f"Reliability specification under the Livingston and Lewis procedure must be a numeric value between 0 and 1 (0 < reliability < 1). Current input is '{lines[0][1]}'.")
        raise TypeError(input_error)
    
    if lines[0][0].lower() != "ll" and 0 > lines[0][1] >= 1:
        print(f"Reliability specification under the Hanson and Brennan procedure must be a numeric value between 0 and 1 (0 <= reliability < 1). Current input is '{lines[0][1]}'.")
        raise TypeError(input_error)

    if lines[0][2] not in [2, 2.0, 4, 4.0]:
        print(f"Model specification must be either 2 or 4. Current input is '{lines[0][2]}'.")
        raise TypeError(input_error)
    
    if len(lines[0]) > 3 and not isinstance(lines[0][3], (float, int)):
        print(f"The fourth entry in the first line of the input file must be an integer. Current input is '{lines[0][3]}'.")
        lines[0] = lines[0][:3]
    if len(lines[0]) == 3:
        print("Setting default minimum expected value for model-fit testing to 0.")
        lines[0].append(0)

    # Line 2.
    if not isinstance(lines[1][0], str):
        print(f"Specification of the file containing the data must be either the absolute or relative path to the data-file. Current input is '{lines[1][0]}")
        raise TypeError(input_error)
    
    if lines[1][1].lower() not in ["r", "f"]:
        print(f"Data-type specification must be either R for raw-scores, or F for frequency distribution of raw-scores. Current input is '{lines[1][1]}'.")
        raise TypeError(input_error)
    
    if lines[1][1].lower() == "r" and lines[0][0].lower() == "ll":
        if len(lines[1]) == 4:
            print("Warning: The LL procedure requires the specification of a minimum possible test-score.")
            print(" No minimum possible test-score was specified as part of the input.")
            print(" Minimum possible test-score will be assumed to be 0.")
            print(" If the actual minimum possible test-score is not 0, specify the minimum possible value as the fifth value of the second line of the input file.")
            lines[1].append(0)

    if lines[1][1].lower() == "f":
        if len(lines[1]) < 4:
            print("When a frequency distribution is specified as data-input, the columns of the data-file representing raw-scores and frequencies must be specified as the third and fourth values of the second line of the input-file.")
            raise TypeError(input_error)
        if not isinstance(lines[1][2], (float, int)) or not isinstance(lines[1][3], (float, int)):
            print(f"When a frequency distribution is specified as data-input, the third and fourth entries in the second line of the input file must be integers. Current input is '{lines[1][2]}' and {lines[1][3]}.")
            raise TypeError(input_error)
        if any(i % 1 != 0 for i in [lines[1][2], lines[1][3]]):
            print(f"When a frequency distribution is specified as data-input, the third and fourth entries in the second line of the input file must be integers. Current input is '{lines[1][2]}' and {lines[1][3]}.")
            raise TypeError(input_error)
    
    # Line 3.
    if not isinstance(lines[2][0], (float, int)):
        print(f"All entries in the third line of the input file must be numeric. Current input is {[lines[2]]}.")
        raise TypeError(input_error)
    if len(lines[2]) == 3 and (not all(isinstance(i, float) for i in lines[2][2]) or not all(0 > i > 1 for i in lines[2][2])):
        print(f"All true-score cut-points must be floating point values between 0 and 1 (0 < cut-point < 1). Current input is {lines[2][2]}.")
        raise TypeError(input_error)
    if len(lines[2]) == lines[2][0]:
        lines[2] = [lines[2][0], lines[2][1:]]
    elif len(lines[2]) == lines[2][0]*2 - 1:
        lines[2] = [lines[2][0], 
                    [lines[2][i] for i in range(1, int(lines[2][0]))],
                    [lines[2][i] for i in range(int(lines[2][0]), len(lines[2]))]]
    else:
        print("Number of specified cut-points do not match the number of specified categories.")
        raise TypeError(input_error)
    return lines


def read_and_parse_data(parsed_input: list) -> tuple:
    datafile: str = parsed_input[1][0].removeprefix('"').removesuffix('"')
    if not os.path.isabs(datafile):
        datafile = os.path.join(os.path.dirname(sys.executable), datafile)
    try:
        with open(datafile, "r") as file:
            data = file.readlines()
    except:
        raise ImportError("Error reading data file. Check whether the file-path is correctly specified.")
    return data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ui.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    parsed_input = read_and_parse_input(input_file)
    data = read_and_parse_data(parsed_input)
    # Continue with the rest of the program logic