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
    if lines[0][0].lower() not in ["ll", "hb"]:
        print(f"Procedure specification must be either LL for the Livingston and Lewis procedure, or HB for the Hanson and Brennan procedure. Current input is '{lines[0][0]}'.")
        raise TypeError(input_error)
    if not isinstance(lines[0][1], (float, int)) or 0 > lines[0][1] > 1:
        print(f"Reliability specification must be a numeric value between 0 and 1.")
        raise TypeError(input_error)
    if len(lines[0]) > 3 and not isinstance(lines[3], (float, int)):
        print("The fourth entry in the first line of the input file must be an integer.")
        lines[0] = lines[0][:3]
    if len(lines[0]) == 3:
        print("Setting default minimum expected value for model-fit testing to 0.")
        lines[0].append(0)
    if not isinstance(lines[1][0], str):
        print("Specification of the file containing the data must be either the absolute or relative path to the data-file.")
        raise TypeError(input_error)
    if lines[1][1].lower() not in ["r", "f"]:
        print(f"Data-type specification must be either R for raw-scores, or F for frequency distribution of raw-scores. Current input is '{lines[1][1]}'.")
        raise TypeError(input_error)
    if lines[1][1].lower() == "r" and lines[0][0].lower() == "ll":
        if len(lines[1]) == 4:
            print("Warning: The LL procedure requires the specification of a minimum possible test-score.")
            print(" No minimum possible test-score was specified as part of the input.")
            print(" Minimum possible test-score will be assumed to be 0.")
            print(" If the minimum possible test-score is not 0, specify the minimum possible value as the fifth value of the second line of the input file.")
            lines[1].append(0)
    if lines[1][1].lower() == "f":
        if len(lines[1]) < 4:
            print("When a frequency distribution is specified as data-input, the columns of the data-file representing raw-scores and frequencies must be specified as the third and fourth values of the second line in the input-file.")
            raise TypeError(input_error)
        if not isinstance(lines[1][2], (float, int)) or not isinstance(lines[1][3], (float, int)):
            print("When a frequency distribution is specified as data-input, the third and fourth entries in the second line of the input file must be integers.")
            raise TypeError(input_error)
        if any(i % 1 != 0 for i in [lines[1][2], lines[1][3]]):
            print("When a frequency distribution is specified as data-input, the third and fourth entries in the second line of the input file must be integers.")
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
    print(" Estimating model parameters.")
    output = bbclassify.bbclassify(data = data[0],
                                   reliability = reliability,
                                   min_score = min_score,
                                   max_score = max_score,
                                   cut_scores = cut_scores,
                                   cut_truescores = cut_truescores,
                                   method = method,
                                   model = model)
    print(" Estimating model fit.")
    output.modelfit(minimum_expected_value = minimum_expected_value)
    print(" Estimating classification accuracy.")
    output.accuracy()
    print(" Estimating classification consistency.")
    output.consistency()

if __name__ == "__main__":
    main()



