import os
import sys
import numpy as np
import pandas as pd
import bbclassify
import statistics as stats
import scipy.stats
def read_and_parse_input(filename: str) -> list:
    input_error = "Input error. Execution terminated."
    #if not os.path.isabs(filename):
    #    filename = os.path.join(os.path.abspath(__file__), filename)
    #try:
    with open (filename, "r") as file:
           lines = file.readlines()
    #except:
    #    raise ImportError("Error reading input file. Check whether the file-path is correctly specified.")
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
    #if not isinstance(lines[2][0], (float, int)):
    #    print(f"All entries in the third line of the input file must be numeric. Current input is {[lines[2]]}.")
    #    raise TypeError(input_error)
    #if len(lines[2]) == 3 and (not all(isinstance(i, float) for i in lines[2][2]) or not all(0 > i > 1 for i in lines[2][2])):
    #    print(f"All true-score cut-points must be floating point values between 0 and 1 (0 < cut-point < 1). Current input is {lines[2][2]}.")
    #    raise TypeError(input_error)
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


def read_and_parse_data(parsed_input: str) -> list:
    datafile: str = parsed_input[1][0].removeprefix('"').removesuffix('"')
    if not os.path.isabs(datafile):
        datafile = os.path.join(os.path.abspath(__file__), datafile)
    datafile = datafile.replace("/ui.py", "")
    with open (datafile, 'r') as file:
        datalines = file.readlines()
    if parsed_input[1][1].lower() == "r":
        data = [float(i[0]) if float(i[0] % 1 != 0) else int(i[0]) for i in datalines]
        datalines = None
    elif parsed_input[1][1].lower() == "f":
        datalines = [[float(k) if float(k) % 1 != 0 else int(k) for k in "".join(i if i.isdigit() or i == "." else " " for i in j).split(" ")[:2]] for j in datalines]
        xcol = int(parsed_input[1][2] - 1)
        fcol = int(parsed_input[1][3] - 1)
        data_full = [[i[xcol] for _ in range(i[fcol])] for i in datalines]
        data = [i for j in data_full for i in j]   
    #non_numeric = [i for i in data[0] if not isinstance(i, (float, int))]
    #len_non_numeric = len(non_numeric)
    #if len_non_numeric != 0:
    #    print(f"All values contained in the data-file must be numeric. Data-file contained {len_non_numeric} non-numeric {"entries" if len_non_numeric > 1 else "entry"}.")
    #    print(f"Ensure that the data-file contains no non-numeric entries (e.g., no column-names).")
    #    if len_non_numeric < 7:
    #        print(f"The data-file contained the following non-numeric {"entries" if len_non_numeric > 1 else "entry"}: {non_numeric}.")
    #    else:
    #        print(f"The first six non-numeric entries encountered in the data-file were: {non_numeric[:6]}.")
    #    raise TypeError("Input error. Execution terminated.")
    return data, datalines


def float_to_str(x: float) -> str:
    x = str(round(x, 5))
    if "." not in x and len(x) < 7:
        x = x + "." + "0"*(7 - len(x) -1)
    elif "." in x and len(x) < 7:
        x = x + "0"*(7 - len(x))
    elif "." in x and len(x) > 7 and x.index(".") < 7:
        x = x[:7]
    elif "." in x and x.index(".") >= 7:
        x = x[:x.index(".")]
    return x
    

def array_to_strlist(x: np.array):
    x = np.hstack((x, x.sum(axis = 1).reshape(-1, 1)))
    x = np.vstack((x, x.sum(axis = 0)))
    strlist = [[round(float(i), 5) if i > 0.0001 else 0 for i in x[j, ]] for j in range(len(x))]
    for i in range(len(x)):
        for j in range(len(x)):
            strlist[i][j] = float_to_str(strlist[i][j])
    return strlist

def add_labels(x, col, row):
    col = [""] + [col + str(i) if i < len(x) - 1 else "marg" for i in range(len(x))]
    col = [" "*(7 - len(i)) + i for i in col]
    row = [row + str(i) if i < len(x) - 1 else "marg" for i in range(len(x))]
    row = [" "*(7 - len(i)) + i for i in row]
    x = [[row[i]] + x[i] for i in range(len(row))]
    x.insert(0, col)
    return x

def main():
    input_file: str = input("Enter path to- or name of the input file: ")
    input_file_name = input_file[::-1].split("/")[0][::-1]
    input_file: list = read_and_parse_input(input_file)

    data = read_and_parse_data(input_file)
    n_observations = len(data[0])
    n_categories = int(input_file[2][0])
    mean = stats.mean(data[0])
    variance = stats.variance(data[0])
    skewness = scipy.stats.skew(data[0])
    kurtosis = scipy.stats.kurtosis(data[0], fisher = False)
    reliability = input_file[0][1]
    if input_file[1][1].lower() == "r":
        min_score = input_file[1][4] if input_file[0][0].lower() == "ll" else 0
        max_score = input_file[1][3]
    if input_file[1][1].lower() == "f":
        min_score = data[1][0][0]
        max_score = data[1][len(data[1]) - 1][0]
    cut_scores = [float(i) if float(i) % 1 != 0 else int(i) for i in input_file[2][1]]
    if len(input_file[2]) == 3:
        cut_truescores = input_file[2][2]
    else:
        cut_truescores = None
    method = input_file[0][0]
    model = input_file[0][2]
    minimum_expected_value = input_file[0][3]
    
    print("PROGRESS:")
    print(" Estimating model parameters...", end = "\r")
    output = bbclassify.bbclassify(data = data[0],
                                    reliability = reliability,
                                    min_score = min_score,
                                    max_score = max_score,
                                    cut_scores = cut_scores,
                                    cut_truescores = cut_truescores,
                                    method = method,
                                    model = model, 
                                    failsafe = True)
    ts_raw_moments = output._tsm(data[0], output.max_score if method.lower() != "ll" else output.effective_test_length, output.Parameters["lords k"])
    ts_moments = [ts_raw_moments[0], ts_raw_moments[1] - ts_raw_moments[0]**2]
    ts_moments.append((ts_raw_moments[2] - 3*(ts_raw_moments[0]*ts_raw_moments[1]) + 2*ts_raw_moments[0]**3) / (ts_moments[1]**.5)**3)
    ts_moments.append((ts_raw_moments[3] - 4*(ts_raw_moments[0] * ts_raw_moments[2]) + 6*(ts_raw_moments[0]**2 * ts_raw_moments[1]) - 3*ts_raw_moments[0]**4) / (ts_moments[1]**.5)**4)
    print(" Estimating model parameters... \033[92m✓\033[0m")
    print(" Estimating model fit...", end = "\r")
    output.modelfit(minimum_expected_value = minimum_expected_value)
    print(" Estimating model fit... \033[92m✓\033[0m")
    print(" Estimating classification accuracy...", end = "\r")
    output.accuracy()
    print(" Estimating classification accuracy... \033[92m✓\033[0m")    
    rounded_confusionmatrix = add_labels(array_to_strlist(output.confusionmatrix), "x", "t")
    tp, tn, fp, fn, sensitivity, specificity = [], [], [], [], [], []
    for i in range(n_categories):
        tp.append(output.confusionmatrix[i, i])
        fp.append(output.confusionmatrix[:, i].sum() - tp[i])
        fn.append(output.confusionmatrix[i, :].sum() - tp[i])
        tn.append(1 - (tp[i] + fp[i] + fn[i]))
        sensitivity.append(tp[i] / (tp[i] + fn[i]))
        specificity.append(tn[i] / (tn[i] + fp[i]))
    category_proportions = []
    for i in range(n_categories):
        if i == 0:
            category_proportions.append(len([i for i in data[0] if i < input_file[2][1][0]]) / n_observations)
        elif 0 < i < (n_categories - 1):
            category_proportions.append(len([j for j in data[0] if j >= cut_scores[i - 1] and j < cut_scores[i]]) / n_observations)
        else:
            category_proportions.append(len([i for i in data[0] if i >= cut_scores[len(cut_scores) - 1]]) / n_observations)
    print(" Estimating classification consistency... ", end = "\r")
    output.consistency()
    print(" Estimating classification consistency... \033[92m✓\033[0m")
    rounded_consistencymatrix = add_labels(array_to_strlist(output.consistencymatrix), "x", "x")
    chance_consistency = sum([sum(i)**2 for i in output.consistencymatrix])
    coefficient_kappa = (output.Consistency - chance_consistency) / (1 - chance_consistency)

    with open("BBClassify_output", "w") as file:

        file.write("******************************************************************************\n")
        file.write("***   BBClassify:  Beta-Binomial Classification Accuracy and Consistency   ***\n")
        file.write("***                              Version 1.0.0                             ***\n")
        file.write("***                                                                        ***\n")
        file.write("***                           Haakon E. Haakstad                           ***\n")
        file.write("***                                                                        ***\n")
        file.write("***            MIT License (https://opensource.org/license/mit)            ***\n")
        file.write("***                Copyright (c) 2025 Haakon Eidem Haakstad                ***\n")
        file.write("******************************************************************************\n")
        file.write("\n")
        file.write(f"*** Listing of Input Specified in \"{input_file_name}\" ***\n")
        file.write("\n")
        file.write(f" Type of Procedure:           {"Livingston and Lewis ('LL')." if method.lower() == "ll" else "Hanson and Brennan ('HB').\n"}")
        file.write(f" Reliability of scores:       {reliability}\n")
        file.write(f" True-score Beta model:       {int(model)}-parameter Beta distribution.\n")
        file.write(f" Model-fit testing:           Minimum expected value of bins set to {minimum_expected_value}.\n")
        file.write(f" Name of data file:           {input_file[1][0]}\n")
        file.write(f" Format of input data:        {"Raw scores" if input_file[1][1].lower() == "r" else "Frequency distribution of raw scores"}.\n")
        file.write(f" Maximum possible score:      {max_score}\n")
        file.write(f" Minimum possible score:      {min_score}\n")
        file.write(f" Number of categories:        {int(input_file[2][0])}\n")
        file.write(f" Obs.-score cut-point(s):     {", ".join([str(i) for i in cut_scores])}\n")
        file.write(f" True-score cut-point(s):     {", ".join([str((i - min_score) / (max_score - min_score)) for i in cut_scores] if cut_truescores is None else [str(i) for i in cut_truescores])}\n")
        file.write("\n")
        file.write("\n")
        file.write(f"*** Summary Statistics of Data in {input_file[1][0]} ***\n")
        file.write("\n")
        file.write(f" Number of observations:      {n_observations}\n")
        file.write("\n")
        file.write(" Observed-score distribution moments:\n")
        file.write(f"  Mean:                       {float_to_str(mean)}\n")
        file.write(f"  Variance:                   {float_to_str(variance)} (SD = {float_to_str(variance**.5)})\n")
        file.write(f"  Skewness:                   {float_to_str(skewness)}\n")
        file.write(f"  Kurtosis:                   {float_to_str(kurtosis)}\n")
        file.write("\n")
        file.write(" Observed category proportions:\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:                 {float_to_str(category_proportions[i])}\n")
        file.write("\n")
        file.write("\n")
        file.write("*** Model Parameter Estimates ***\n")
        file.write("\n")
        if output.failsafe_engaged:
            file.write(" WARNING: Four-parameter fitting procedure produced impermissible location\n")
            file.write("  parameter estimates. Reverted to a two-parameter fitting procedure with\n")
            file.write("  location parameters l = 0 and u = 1.\n")
            file.write("\n")
        file.write(f" Proportional true-score distribution moments:\n")
        file.write(f"  Mean:                       {float_to_str(ts_moments[0])}\n")
        file.write(f"  Variance:                   {float_to_str(ts_moments[1])} (SD = {float_to_str(ts_moments[1]**.5)})\n")
        file.write(f"  Skewness:                   {float_to_str(ts_moments[2])}\n")
        file.write(f"  Kurtosis:                   {float_to_str(ts_moments[3])}\n")
        file.write("\n")
        file.write(f" Reliability:                 {float_to_str((ts_moments[1]**.5 * max_score)**2 / variance)}\n")
        file.write("\n")
        file.write(f" Number of moments fit:       {int(output.model)} ({"Mean and Variance" if output.model == 2 else "Mean, Variance, Skewness, and Kurtosis"})\n")
        file.write("\n")
        file.write(f" Beta true-score distribution:\n")
        file.write(f"  Alpha:                      {float_to_str(output.Parameters["alpha"])}\n")
        file.write(f"  Beta:                       {float_to_str(output.Parameters["beta"])}\n")
        file.write(f"  l:                          {float_to_str(output.Parameters["l"])}\n")
        file.write(f"  u:                          {float_to_str(output.Parameters["u"])}\n")
        file.write("\n")
        file.write(f" Binomial error distribution:\n")
        file.write(f"  Lord's k:                   {output.Parameters["lords k"]} ({"Compound-Binomial error model" if output.Parameters["lords k"] != 0 else "Binomial error model"})\n")
        file.write(f"  Number of 'trials':         {output.N} ({"Effective Test Length" if method.lower() == "ll" else "Actual Test Length"})\n")
        file.write("\n")
        file.write("\n")
        file.write("*** Model Fit ***\n")
        file.write(f" Pearson's \u03C7\u00B2:                {float_to_str(output.Modelfit_chi_squared)}\n")
        file.write(f" DF:                          {int(output.Modelfit_degrees_of_freedom)}\n")
        file.write(f" p-value:                     {float_to_str(output.Modelfit_p_value)}\n")
        file.write("\n")
        file.write("\n")
        file.write("*** Classification Accuracy Estimates ***\n")
        file.write("\n")
        file.write(" Confusion matrix:\n")
        for i in range(len(rounded_confusionmatrix)): file.write(f"{"   ".join(rounded_confusionmatrix[i])}\n")
        file.write("\n")
        file.write(f" Overall:\n")
        file.write(f"  Accuracy:                   {float_to_str(output.Accuracy)}\n")
        file.write(f"  Sensitivity:                {float_to_str(sum([sensitivity[i] * category_proportions[i] for i in range(n_categories)]))} (weighted average)\n")
        file.write(f"  Specificity:                {float_to_str(sum([specificity[i] * category_proportions[i] for i in range(n_categories)]))} (weighted average)\n")
        file.write("\n")
        file.write(" Category specific:\n")
        file.write("\n")
        file.write("\n")
        file.write("*** Classification Consistency Estimates ***\n")
        file.write("\n")
        file.write(" Consistency matrix:\n")
        for i in range(len(rounded_consistencymatrix)): file.write(f"{"   ".join(rounded_consistencymatrix[i])}\n")
        file.write("\n")
        file.write(f" Overall:\n") 
        file.write(f"  Consistency:                {float_to_str(output.Consistency)}\n")
        file.write(f"  Chance consistency:         {float_to_str(chance_consistency)}\n")
        file.write(f"  Coefficient Kappa:          {float_to_str(coefficient_kappa)}\n")
        file.write("\n")
        file.write(" Category specific:\n")

if __name__ == "__main__":
    main()