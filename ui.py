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
    if len(lines[0]) == 3:
        lines[0].append(0)
    if lines[1][1].lower() == "r" and lines[0][0].lower() == "ll":
        if len(lines[1]) == 4:
            print("""Warning: The LL procedure requires the specification of a minimum possible test-score.\n
                          No minimum possible test-score was specified as part of the input.\n
                          Minimum possible test-score is assumed to be 0.""")
            lines[1].append(0)
    if len(lines[2]) == lines[2][1]:
        lines[2] = [lines[2][0], [lines[2][1:]]]
    if len(lines[2]) == lines[2][0]*2 - 1:
        lines[2] = lines[2] = [lines[2][0], lines[2][1:lines[2][0]], lines[2][lines[2][0]:]]
    return lines

def read_and_parse_data(input_file: str) -> list:
    datafile: str = input_file[1][0].removeprefix('"').removesuffix('"')
    if not os.path.isabs(datafile):
        datafile = os.path.join(os.path.abspath(__file__), datafile)
    with open (datafile, 'r') as file:
        datalines = file.readlines()
    if input_file[1][1].lower() == "r":
        data = [i[0] for i in datalines]
    elif input_file[1][1].lower() == "f":
        data = [[i[0] for _ in range(i[1])] for i in datalines]
        data = [i for j in data for i in j]
    return data

trying = os.path.abspath(__file__)[::-1]
trying[trying.index("/") - 1:][::-1]
print(read_and_parse_input(trying[trying.index("/"):][::-1] + "ui_test/test_input"))

def main():
    input_file: list = input("Enter path to- or name of the input file.")
    input_file: list = read_and_parse_input(input_file)
    data = read_and_parse_data(input_file)
    reliability = input_file[0][1]
    if input_file[1][1].lower() == "m":
        min_score = input_file[1][4]
        max_score = input_file[1][3]
    if input_file[1][1].lower() == "f":
        min_score = input_file[1][5]
        max_score = input_file[1][4]
    cut_scores = input_file[2][1]
    if len(input_file[2]) == 3:
        cut_truescores = input_file[2][2]
    else:
        cut_truescores = None
    method = input_file[0][0]
    model = input_file[0][2]
    minimum_expected_value = input_file[0][3]

    print("INPUT:")
    print(f" Procedure:         {"Livingston and Lewis ('LL')." if method == "ll" else "Hanson and Brennan ('HB')."}")
    print(f" Reliability:       {reliability}.")
    print(f" True-score model:  {model}-parameter Beta distribution.")
    print(f" Model-fit testing: {1}")

    output = bbclassify.bbclassify(data = data,
                                   reliability = reliability,
                                   min_score = min_score,
                                   max_score = max_score,
                                   cut_scores = cut_scores,
                                   cut_truescores = cut_truescores,
                                   method = method,
                                   model = model)
    output.modelfit(minimum_expected_value = minimum_expected_value)
    output.accuracy()
    output.consistency()

if __name__ == "__main__":
    main()



