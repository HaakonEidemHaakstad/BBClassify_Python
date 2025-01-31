import os
import sys
import numpy as np

def read_and_parse_input(filename: str, raw: bool = False, compile: bool = False) -> list:
    input_error: str = "Input error. Execution terminated."
    
    if not os.path.isabs(filename):
        if compile:
            # If the script is compiled, use the executable's directory
            base_path = os.path.dirname(sys.executable)
        else:
            # If the script is not compiled, use the script's directory
            base_path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_path, filename)

    filename = filename.replace("support_functions.py/", "")
    filename = filename.replace("support_functions.py\\", "")
    filename = filename.replace("ui.py/", "")
    filename = filename.replace("ui.py\\", "")
    try:
        with open(filename, "r") as file:
            lines: list[str] = file.readlines()
    except:
        e = filename[::-1]
        e1 = e[:e.index("/" if "/" in e else "\\")][::-1]
        e = e[::-1].replace(e1, "")
        print(f"Input file \"{e1}\" not found at \"{e}\".\n")
        raise FileNotFoundError(input_error)
    if raw: 
        return lines
    
    lines = [i.lower().split() for i in lines]
    lines = [[float(i) if i.replace(".", "", 1).replace("-", "", 1).isdigit() else i for i in j] for j in lines]

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

def read_and_parse_data(parsed_input: list, compile: bool = False) -> tuple:
    datafile: str = parsed_input[1][0].removeprefix('"').removesuffix('"')
    if not os.path.isabs(datafile):
        if compile:
            # If the script is compiled, use the executable's directory
            base_path = os.path.dirname(sys.executable)
        else:
            # If the script is not compiled, use the script's directory
            base_path = os.path.dirname(os.path.abspath(__file__))
        datafile = os.path.join(base_path, datafile)

    datafile = datafile.replace("support_functions.py/", "")
    datafile = datafile.replace("support_functions.py\\", "")
    datafile = datafile.replace("ui.py/", "")
    datafile = datafile.replace("ui.py\\", "")
    try:
        with open (datafile, 'r') as file:
            datalines: list = file.readlines()
    except:
        e = datafile[::-1]
        e1 = e[:e.index("/" if "/" in e else "\\")][::-1]
        e = e[::-1].replace(e1, "")
        print(f"Data file \"{e1}\" not found at \"{e}\".\n")
    if parsed_input[1][1].lower() == "r":
        data: list = [float(i[0]) if float(i[0] % 1 != 0) else int(i[0]) for i in datalines]
        datalines = None
    elif parsed_input[1][1].lower() == "f":
        datalines = [[float(k) if float(k) % 1 != 0 else int(k) for k in "".join(i if i.isdigit() or i == "." else " " for i in j).split(" ")[:2]] for j in datalines]
        xcol: int = int(parsed_input[1][2] - 1)
        fcol: int = int(parsed_input[1][3] - 1)
        data_full: list[list] = [[i[xcol] for _ in range(i[fcol])] for i in datalines]
        data = [i for j in data_full for i in j]
    non_numeric = [i for i in data if not isinstance(i, (float, int))]
    len_non_numeric = len(non_numeric)
    if len_non_numeric != 0:
        print(f"All values contained in the data-file must be numeric. Data-file contained {len_non_numeric} non-numeric {"entries" if len_non_numeric > 1 else "entry"}.")
        print(f"Ensure that the data-file contains no non-numeric entries (e.g., no column-names).")
        if len_non_numeric < 7:
            print(f"The data-file contained the following non-numeric {"entries" if len_non_numeric > 1 else "entry"}: {non_numeric}.")
        else:
            print(f"The first six non-numeric entries encountered in the data-file were: {non_numeric[:6]}.")
        raise TypeError("Input error. Execution terminated.")
    return data, datalines

def float_to_str(x: float) -> str:
    """
    Convert a float to a string with a specific format.

    This function converts a float to a string, rounding it to 5 decimal places.
    It ensures that the resulting string has a length of 7 characters, including
    the decimal point and trailing zeros if necessary.

    Parameters:
    x (float): The float to be converted.

    Returns:
    str: The formatted string representation of the float.
    """
    x: str = str(round(x, 5))
    if "." not in x and len(x) < 7:
        x = x + "." + "0"*(7 - len(x) -1)
    elif "." in x and len(x) < 7:
        x = x + "0"*(7 - len(x))
    elif "." in x and len(x) > 7 and x.index(".") < 7:
        x = x[:7]
    elif "." in x and x.index(".") >= 7:
        x = x[:x.index(".")]
    return x    

def array_to_strlist(x: np.array) -> list[str]:
    """
    Convert a 2D numpy array to a list of strings with formatted numbers.

    This function takes a 2D numpy array, rounds each element to 6 decimal places,
    converts the elements to strings, and adds row and column sums.

    Parameters:
    x (np.array): The 2D numpy array to be converted.

    Returns:
    list[str]: A list of strings representing the formatted 2D array with row and column sums.
    """
    x: list[list] = [[round(j, 6) for j in i] for i in x]
    x = [i + [sum(i)] for i in x]
    x.append([sum(j) for j in [list(i) for i in zip(*x)]])
    x = [[float_to_str(j) for j in i] for i in x]
    return x

def add_labels(x: list[list], col:  int, row: int) -> list:
    """
    Add row and column labels to a 2D list.

    This function adds labels to the rows and columns of a 2D list. The labels
    are generated based on the provided column and row identifiers.

    Parameters:
    x (list[list]): The 2D list to which labels will be added.
    col (int): The identifier for the column labels.
    row (int): The identifier for the row labels.

    Returns:
    list: A 2D list with the added row and column labels.
    """
    col: list = [""] + [col + str(i) if i < len(x) - 1 else "marg" for i in range(len(x))]
    col: list = [" "*(7 - len(i)) + i for i in col]
    row: list = [row + str(i) if i < len(x) - 1 else "marg" for i in range(len(x))]
    row: list = [" "*(7 - len(i)) + i for i in row]
    x: list[list] = [[row[i]] + x[i] for i in range(len(row))]
    x.insert(0, col)
    return x

def factorial_from_ordinary_moments(mean: float, variance: float, skewness: float, kurtosis: float) -> tuple:    
    """
    Calculate factorial moments from ordinary moments.

    This function converts the ordinary moments (mean, variance, skewness, and kurtosis)
    to factorial moments using Stirling numbers of the second kind.

    Parameters:
    mean (float): The mean of the distribution.
    variance (float): The variance of the distribution.
    skewness (float): The skewness of the distribution.
    kurtosis (float): The kurtosis of the distribution.

    Returns:
    tuple: A tuple containing the mean, factorial variance (sigma_2_f), 
           factorial skewness (gamma_3_f), and factorial kurtosis (gamma_4_f).
    """
    sigma_4 = kurtosis * variance**2
    sigma_3 = skewness * variance**(3/2)

    m1 = mean
    m2 = variance + mean**2
    m3 = sigma_3 + 3*mean*variance + mean**3
    m4 = sigma_4 + 4*mean*sigma_3 + 6*mean**2*variance + mean**4

    f2 = m2 - m1
    f3 = m3 - 3*m2 + 2*m1
    f4 = m4 - 6*m3 + 11*m2 - 6*m1

    sigma_2_f = f2 - mean**2
    gamma_3_f = f3 - 3*mean*f2 + 2*mean**3
    gamma_4_f = f4 - 4*mean*f3 + 6*mean**2*f2 - 3*mean**4

    return mean, sigma_2_f, gamma_3_f, gamma_4_f