import os
import sys
import pandas as pd
import bbclassify

def read_and_parse_input(filename: str) -> list:
    input_error = "Input error. Execution terminated."
    if not os.path.isabs(filename):
        filename = os.path.join(os.path.abspath(__file__), filename)
    try:
        with open (filename, "r") as file:
            lines = file.readlines()
    except:
        raise ImportError("Error reading input file. Check whether the file-path is correctly specified.")
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
    
    if lines[0][0].lower() == "ll" and 0 > lines[0][1] >= 1:
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
        print(f"All entries in the third line of the input file must be integers. Current input is {[lines[2]]}.")
        raise TypeError(input_error)
    if len(lines[2]) == 3 and not all(isinstance(i, float) for i in lines[2][2]) or not all(0 > i > 1 for i in lines[2][2]):
        print(f"All true-score cut-points must be floating point values between 0 and 1 (0 < cut-point < 1). Current input is {lines[2][2]}.")
        raise TypeError(input_error)
    if len(lines[2]) == lines[2][1]:
        lines[2] = [lines[2][0], [lines[2][1:]]]
    elif len(lines[2]) == lines[2][0]*2 - 1:
        lines[2] = lines[2] = [lines[2][0], lines[2][1:lines[2][0]], lines[2][lines[2][0]:]]
    else:
        print("Number of specified cut-points do not match the number of specified categories.")
        raise TypeError(input_error)    
    return lines

def read_and_parse_data(input_file: str) -> list:
    datafile: str = input_file[1][0].removeprefix('"').removesuffix('"')
    if not os.path.isabs(datafile):
        datafile = os.path.join(os.path.abspath(__file__), datafile)
    with open (datafile, 'r') as file:
        datalines = file.readlines()
    if input_file[1][1].lower() == "r":
        data = ([i[0] for i in datalines])
    elif input_file[1][1].lower() == "f":
        xcol = input_file[1][2] - 1
        fcol = input_file[1][3] - 1
        data_full = [[i[xcol] for _ in range(i[fcol])] for i in datalines]
        data = ([i for j in data for i in j], data_full)    
    non_numeric = [i for i in data[0] if not isinstance(i, (float, int))]
    len_non_numeric = len(non_numeric)
    if len_non_numeric != 0:
        print(f"All values contained in the data-file must be numeric. Data-file contained {len_non_numeric} non-numeric {"entries" if len_non_numeric > 1 else "entry"}.")
        print(f"Ensure that the data-file contains no non-numeric entries (e.g., no column-names).")
        if len_non_numeric < 7:
            print(f"The data-file contained the following non-numeric {"entries" if len_non_numeric > 1 else "entry"}: {non_numeric}.")
        else:
            print(f"The first six non-numeric entries encountered in the data-file were: {non_numeric[:6]}.")
        raise TypeError("Input error. Execution terminated.")
    return data

trying = os.path.abspath(__file__)[::-1]
trying[trying.index("/") - 1:][::-1]
print(read_and_parse_input(trying[trying.index("/"):][::-1] + "ui_test/test_input"))

def main():
    input_file: list = input("Enter path to- or name of the input file.")
    input_file: list = read_and_parse_input(input_file)
    data = read_and_parse_data(input_file)
    reliability = input_file[0][1]
    if input_file[1][1].lower() == "r":
        min_score = input_file[1][4] if input_file[0][0].lower() == "ll" else 0
        max_score = input_file[1][3]
    if input_file[1][1].lower() == "f":
        min_score = data[1][0][input_file[1][2] - 1]
        max_score = data[1][len(data[1]) - 1][input_file[1][2] - 1]
    cut_scores = input_file[2][1]
    if len(input_file[2]) == 3:
        cut_truescores = input_file[2][2]
    else:
        cut_truescores = None
    method = input_file[0][0]
    model = input_file[0][2]
    minimum_expected_value = input_file[0][3]

    print("INPUT:")
    print(f" Type of Procedure:     {"Livingston and Lewis ('LL')." if method.lower() == "ll" else "Hanson and Brennan ('HB')."}")
    print(f" Reliability of scores: {reliability}.")
    print(f" True-score Beta model: {model}-parameter Beta distribution.")
    print(f" Model-fit testing:     Minimum expected value of bins set to {minimum_expected_value}.")
    print(f" Name of data file      {input_file[1][0]}")
    print(f" Format of input data:  {"Raw scores" if input_file[1][1].lower() == "r" else "Frequency distribution of raw scores"}.")
    print(f" Max. possible score:   {max_score}.")
    print(f" Min. possible score:   {min_score}.")
    print(f" Number of categories:  {input_file[2][0]}.")
    print(f" Obs. score cut-points: {input_file[2][1]}.")
    print(f" True-score cut-points: {"None specified" if cut_truescores is None else cut_truescores}.")
    
    print("\n\n")

    print("PROGRESS:")
    print(" Estimating model parameters...", end = "\r")
    output = bbclassify.bbclassify(data = data[0],
                                    reliability = reliability,
                                    min_score = min_score,
                                    max_score = max_score,
                                    cut_scores = cut_scores,
                                    cut_truescores = cut_truescores,
                                    method = method,
                                    model = model)
    print(" Estimating model parameters... \033[92m✓\033[0m")
    print(" Estimating model fit...", end = "\r")
    output.modelfit(minimum_expected_value = minimum_expected_value)
    print(" Estimating model fit... \033[92m✓\033[0m")
    print(" Estimating classification accuracy...", end = "\r")
    output.accuracy()
    print(" Estimating classification accuracy... \033[92m✓\033[0m")
    print(" Estimating classification consistency... ", end = "\r")
    output.consistency()
    print(" Estimating classification consistency... \033[92m✓\033[0m")

    with open("BBClassify_output", "w") as file:
        file.write("*************************************************************************\n")
        file.write("*** BBClassify: Beta-Binomial Classification Accuracy and Consistency ***\n")
        file.write("***                            Version 1.0.0                          ***\n")
        file.write("***                                                                   ***\n")
        file.write("***                         Haakon E. Haakstad                        ***\n")
        file.write("***                            Unaffiliated                           ***\n")
        file.write("***                                                                   ***\n")
        file.write("***          MIT License (https://opensource.org/license/mit)         ***\n")
        file.write("***              Copyright (c) 2025 Haakon Eidem Haakstad             ***\n")
        file.write("*************************************************************************\n")
        file.write("\n")
        file.write(f"*** {"Livingston and Lewis" if method.lower() == "ll" else "Hanson and Brennan"} Results ***\n")
        file.write("\n")
        file.write(f"*** Listing of Input Specified in the Input File \"{input_file[1][0]}\" ***")
        file.write("\n")
        file.write(f" Type of Procedure:     {"Livingston and Lewis ('LL')." if method.lower() == "ll" else "Hanson and Brennan ('HB').\n"}")
        file.write(f" Reliability of scores: {reliability}\n")
        file.write(f" True-score Beta model: {model}-parameter Beta distribution.\n")
        file.write(f" Model-fit testing:     Minimum expected value of bins set to {minimum_expected_value}.\n")
        file.write(f" Name of data file      {input_file[1][0]}\n")
        file.write(f" Format of input data:  {"Raw scores" if input_file[1][1].lower() == "r" else "Frequency distribution of raw scores"}.\n")
        file.write(f" Max. possible score:   {max_score}\n")
        file.write(f" Min. possible score:   {min_score}\n")
        file.write(f" Number of categories:  {input_file[2][0]}\n")
        file.write(f" Obs. score cut-points: {input_file[2][1]}\n")
        file.write(f" True-score cut-points: {"None specified" if cut_truescores is None else cut_truescores}.\n")
        file.write("\n")
        file.write("*** Model Parameter Estimates ***\n")
        file.write(f"Beta true-score distribution:\n")
        file.write(f" Alpha: {output.Parameters["alpha"]}\n")
        file.write(f" Beta: {output.Parameters["beta"]}\n")
        file.write(f" Lower-bound: {output.Parameters["l"]}\n")
        file.write(f" Upper-bound: {output.Parameters["u"]}\n")
        file.write("\n")
        file.write(f"Binomial error distribution:\n")
        file.write(f" Lord's k: {output.Parameters["lords k"]} ({"Compound-Binomial error model" if output.Parameters["lords k"] != 0 else "Binomial error model"})\n")
        file.write(f" Number of 'trials': {output.N} ({"Effective Test Length" if method.lower() == "ll" else "Actual Test Length"})\n")

if __name__ == "__main__":
    main()