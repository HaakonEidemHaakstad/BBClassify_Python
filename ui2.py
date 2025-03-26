#!/usr/bin/env python3
"""
UI2.py - BBClassify: Beta-Binomial Classification Accuracy and Consistency
Revised complete version with robust input parsing, improved handling of cut scores,
safeguarded calculations for sensitivity/specificity, and transposed confusion matrix output.

NOTE: If using raw data input, ensure that the control card's maximum score is at least
      as high as the highest observed score. This version filters the raw data to include
      only numeric values within the expected range.
"""

import os
import sys
import time
import threading
import shlex

# Display welcome message.
print("WELCOME TO BBClassify: BETA-BINOMIAL CLASSIFICATION ACCURACY AND CONSISTENCY")

# ------------------------------------------------------------------------------
# Helper Functions for Tokenization and Conversion
# ------------------------------------------------------------------------------

def tokenize_line(line):
    """Tokenizes a line using shlex.split to respect quoted strings."""
    return shlex.split(line)

def convert_token(token):
    """
    Attempts to convert a token to a float or int.
    Returns a float if the token contains a decimal point,
    an int if it is numeric without a decimal point,
    otherwise returns the original token.
    """
    try:
        if "." in token:
            return float(token)
        elif token.isnumeric():
            return int(token)
    except Exception:
        pass
    return token

def convert_tokens(tokens):
    """Applies convert_token to each token in a list."""
    return [convert_token(token) for token in tokens]

# ------------------------------------------------------------------------------
# Control Card Validation
# ------------------------------------------------------------------------------

def validate_control_card(parsed_input):
    """
    Validates the control card (input file) contents.
    
    Returns a dictionary with keys:
      - errors: List of error messages
      - warnings: List of warning messages
      - notes: List of informational notes
      - parsed_input: The tokenized control card lines
      - parameters: Extracted parameters as a dictionary
    """
    errors = []
    warnings_list = []
    notes = []
    
    if len(parsed_input) < 3:
        errors.append("Input file must contain at least 3 lines.")
    
    # Line 1: Procedure and model parameters.
    if len(parsed_input[0]) < 3:
        errors.append("First line must contain at least 3 values.")
    if len(parsed_input[0]) == 3:
        parsed_input[0].append("0")
        notes.append("Fourth value of the first line not specified. Defaulting to 0.")
    
    if len(parsed_input[0]) > 0:
        proc = parsed_input[0][0].lower()
        if proc not in ["hb", "ll"]:
            errors.append("The first value must be either 'hb' or 'll'.")
    else:
        proc = ""
    
    if len(parsed_input[0]) > 1:
        rel = parsed_input[0][1]
        try:
            rel = float(rel)
        except ValueError:
            errors.append("Reliability must be numeric.")
        else:
            if rel != -1 and (rel < 0 or rel > 1):
                errors.append("Reliability must be -1 (for complete data) or between 0 and 1.")
            if proc == "ll" and (rel <= 0 or rel >= 1):
                errors.append("Under LL, reliability must be > 0 and < 1.")
            if proc == "hb" and (rel < 0 or rel >= 1):
                errors.append("Under HB, reliability must be between 0 (inclusive) and 1 (exclusive).")
    else:
        rel = None
    
    if len(parsed_input[0]) > 2:
        mod = parsed_input[0][2]
        try:
            mod = int(mod)
        except ValueError:
            errors.append("Model parameter must be an integer (2 or 4).")
        else:
            if mod not in [2, 4]:
                errors.append("Model parameter must be either 2 or 4.")
    else:
        mod = None
    
    if len(parsed_input[0]) > 3:
        mev = parsed_input[0][3]
        try:
            mev = int(mev)
        except ValueError:
            warnings_list.append("Minimum expected value must be an integer. Defaulting to 0.")
            mev = 0
        else:
            if mev < 0:
                warnings_list.append("Minimum expected value must be >= 0. Defaulting to 0.")
                mev = 0
    else:
        mev = 0

    # Line 2: Data file path, data type, and additional parameters.
    if len(parsed_input[1]) < 2:
        errors.append("The second line must contain at least 2 values.")
    data_file_path = parsed_input[1][0].removeprefix("\"").removesuffix("\"")
    if not os.path.isabs(data_file_path):
        file_name = data_file_path
        base_path = os.path.dirname(sys.executable)
        data_file_path = os.path.join(base_path, data_file_path)
    if not os.path.exists(data_file_path):
        errors.append("Data file '{}' not found.".format(data_file_path))
    
    dt = parsed_input[1][1].lower()
    if dt not in ["r", "f", "m", "c"]:
        errors.append("Data type must be 'r', 'f', 'm', or 'c'.")
    
    if dt == "r":
        if len(parsed_input[1]) < 4:
            errors.append("For raw data, at least 4 values are required on line 2.")
        if not isinstance(parsed_input[1][2], int):
            errors.append("For raw data, the third value must be an integer (data column index).")
        if proc == "hb":
            if not isinstance(parsed_input[1][3], int):
                errors.append("For HB raw data, the fourth value must be an integer (test length).")
        else:
            if not isinstance(parsed_input[1][3], (float, int)):
                errors.append("For LL raw data, the fourth value must be numeric (maximum possible test score).")
            if len(parsed_input[1]) == 4:
                notes.append("For LL raw data, fifth value not specified. Defaulting to 0.")
                parsed_input[1].append("0")
            if not isinstance(parsed_input[1][4], (float, int)):
                warnings_list.append("For LL raw data, the fifth value must be numeric. Defaulting to 0.")
                parsed_input[1][4] = "0"
    
    elif dt == "f":
        if len(parsed_input[1]) < 4:
            errors.append("For frequency data, at least 4 values are required on line 2.")
        if len(parsed_input[1]) > 2:
            if not (isinstance(parsed_input[1][2], int) or (isinstance(parsed_input[1][2], float) and parsed_input[1][2] % 1 == 0)):
                errors.append("For frequency data, the third value must be an integer (raw-score column).")
        if len(parsed_input[1]) > 3:
            if not (isinstance(parsed_input[1][3], int) or (isinstance(parsed_input[1][3], float) and parsed_input[1][3] % 1 == 0)):
                errors.append("For frequency data, the fourth value must be an integer (frequency column).")
            if parsed_input[1][2] == parsed_input[1][3]:
                errors.append("For frequency data, the third and fourth values must be different columns.")
    
    elif dt == "c":
        if len(parsed_input[1]) > 2:
            if proc == "hb":
                if not isinstance(parsed_input[1][2], int):
                    errors.append("For HB complete data, the third value must be an integer (test length).")
            else:
                if not isinstance(parsed_input[1][2], (float, int)):
                    errors.append("For LL complete data, the third value must be numeric (maximum possible test score).")
                if len(parsed_input[1]) < 4:
                    notes.append("For LL complete data, fourth value not specified. Defaulting to 0.")
                    parsed_input[1].append("0")
                if len(parsed_input[1]) > 3:
                    if not isinstance(parsed_input[1][3], (float, int)):
                        warnings_list.append("For LL complete data, the fourth value must be numeric. Defaulting to 0.")
                        parsed_input[1][3] = "0"
                if float(parsed_input[1][2]) <= float(parsed_input[1][3]):
                    errors.append("For complete data, maximum score must be greater than minimum score.")
    
    elif dt == "m":
        if len(parsed_input[1]) > 2:
            warnings_list.append("Extra values on line 2 are ignored for moment data.")
    
    # Line 3: Cut score specifications.
    if len(parsed_input[2]) < 2:
        errors.append("The third line must contain at least 2 values (cut scores).")
    else:
        if isinstance(parsed_input[2][0], int):
            n_categories = int(parsed_input[2][0])
            if len(parsed_input[2]) != n_categories and len(parsed_input[2]) != n_categories * 2 - 1:
                errors.append("The number of cut-points does not match the specified number of categories.")
    
    return {
        "errors": errors,
        "warnings": warnings_list,
        "notes": notes,
        "parsed_input": parsed_input,
        "parameters": {
            "method": proc,
            "reliability": float(rel) if rel is not None and rel != "0" else rel,
            "model": mod,
            "min_expected_value": mev,
            "data_file_path": data_file_path,
            "datatype": dt,
            "max_score": float(parsed_input[1][3]) if dt in ["r", "c"] and proc == "ll" else (float(parsed_input[1][3]) if dt in ["r", "c"] and proc == "hb" else None),
            "min_score": float(parsed_input[1][4]) if dt in ["r", "c"] and proc == "ll" else (0 if dt in ["r", "c"] and proc == "hb" else None)
        }
    }

# ------------------------------------------------------------------------------
# Data File Validation
# ------------------------------------------------------------------------------

def validate_data_file(data_file_path, method, datatype):
    """
    Validates the contents of the data file.
    For complete ("c") data with the HB procedure, every entry must be 0 or 1.
    
    Returns a list of error messages.
    """
    errors = []
    if datatype == "c" and method == "hb":
        try:
            with open(data_file_path, "r") as df:
                data_contents = df.read().split()
                for value in data_contents:
                    try:
                        int_val = int(value)
                        if int_val not in [0, 1]:
                            errors.append("Data file error: Entry {} is not 0 or 1.".format(value))
                            break
                    except ValueError:
                        errors.append("Data file error: Non-integer value found: {}.".format(value))
                        break
        except Exception as e:
            errors.append("Error reading data file: {}".format(e))
    return errors

# ------------------------------------------------------------------------------
# Main Program
# ------------------------------------------------------------------------------

def main():
    """
    Main function that runs the BBClassify program.
    It validates input, processes the data file, performs model estimation,
    and writes the results to an output file.
    """
    def loading_animation(text):
        """Displays a loading animation until stop_loading is True."""
        while not stop_loading:
            for i in range(1, 4):
                if stop_loading:
                    break
                print(f"{text + '.' * i}\033[0m", end="\r")
                time.sleep(0.25)
            if not stop_loading:
                print(f"{text}   ", end="\r")
        if success:
            print(f"{text}... \033[92m✓\033[0m")
        else:
            print(f"{text}... \033[91m✗\033[0m")
    
    global success, stop_loading
    success = True
    stop_loading = False

    def error(x):
        return "\033[91m" + x + "\033[0m"
    def warning(x):
        return "\033[38;5;214m" + x + "\033[0m"
    def note(x):
        return "\033[92m" + x + "\033[0m"

    print("")
    input_path = input("Enter path to- or name of input file: ")

    errors = []
    warnings_list = []
    notes = []

    if not os.path.isabs(input_path):
        file_name = input_path
        base_path = os.path.dirname(sys.executable)
        input_path = os.path.join(base_path, input_path)

    if not os.path.exists(input_path):
        print("")
        print(f"\033[91mERROR\033[0m: File \"{file_name}\" not found at \"{base_path}\".")
        print("")
        input("Execution terminated. Press ENTER to exit BBClassify.")
        return

    with open(input_path, "r") as file:
        raw_input = file.readlines()

    print("")
    print(f"CONTENTS OF INPUT FILE \"{file_name}\" :")
    print("")
    for line in raw_input:
        print(" " + line.strip())
    print("")

    # Start input validation animation.
    thread = threading.Thread(target=loading_animation, args=("Validating input",))
    thread.start()

    # Tokenize input lines using shlex.split to preserve quoted strings.
    parsed_input = [tokenize_line(line) for line in raw_input]
    parsed_input = [convert_tokens(line) for line in parsed_input]

    validation_results = validate_control_card(parsed_input)
    errors.extend(validation_results["errors"])
    warnings_list.extend(validation_results["warnings"])
    notes.extend(validation_results["notes"])
    params = validation_results["parameters"]

    stop_loading = True
    thread.join()
    stop_loading = False

    # Start data validation animation.
    thread = threading.Thread(target=loading_animation, args=("Validating data",))
    thread.start()

    data_file_errors = validate_data_file(params["data_file_path"], params["method"], params["datatype"])
    errors.extend(data_file_errors)

    stop_loading = True
    thread.join()
    stop_loading = False

    if errors:
        for err in errors:
            print(error(err))
        input("Execution terminated. Press ENTER to exit BBClassify.")
        return
    if warnings_list:
        for warn in warnings_list:
            print(warning(warn))
    if notes:
        for nt in notes:
            print(note(nt))
    
    print("")
    print(note("Input validation completed successfully."))
    print("")

    # Load and process the data file.
    with open(params["data_file_path"], "r") as file:
        data_lines = file.readlines()
    # Tokenize data lines.
    data = [tokenize_line(line) for line in data_lines]
    data = [convert_tokens(line) for line in data]
    
    # Process data based on datatype.
    if params["datatype"].lower() == "f":
        xcol = int(parsed_input[1][2]) - 1
        fcol = int(parsed_input[1][3]) - 1
        raw_scores = [row[xcol] for row in data if len(row) > xcol]
        max_score = int(max(raw_scores))
        min_score = int(min(raw_scores))
        data = [[row[xcol] for _ in range(int(row[fcol]))] for row in data]
        data = [score for sublist in data for score in sublist]
    elif params["datatype"].lower() == "r":
        # Robustly flatten raw data.
        flat_data = []
        for sublist in data:
            for item in sublist:
                try:
                    num = float(item)
                    flat_data.append(num)
                except Exception:
                    tokens = item.split()
                    for token in tokens:
                        try:
                            num = float(token)
                            flat_data.append(num)
                        except Exception:
                            continue
        data = [x for x in flat_data if params["min_score"] <= x <= params["max_score"]]
        max_score = params["max_score"]
        min_score = params["min_score"]
    else:
        max_score = params["max_score"]
        min_score = params["min_score"]

    method = params["method"]
    model = params["model"]
    min_expected_value = params["min_expected_value"]

    # Process cut scores from line 3.
    n_categories = int(parsed_input[2][0])
    if len(parsed_input[2]) == n_categories:
        obs_cut_points = [float(x) for x in parsed_input[2][1:]]
        full_scores = [min_score] + obs_cut_points + [max_score]
        true_cut_points = [(score - min_score) / (max_score - min_score) for score in full_scores][1:-1]
        cut_points = obs_cut_points
    elif len(parsed_input[2]) == (2 * n_categories - 1):
        obs_cut_points = [float(x) for x in parsed_input[2][1:n_categories]]
        true_cut_points = [float(x) for x in parsed_input[2][n_categories:]]
        cut_points = obs_cut_points
    else:
        errors.append("Length of cut-point list does not match the number of categories.")
    if errors:
        for err in errors:
            print(error(err))
        input("Execution terminated. Press ENTER to exit BBClassify.")
        return

    # Perform data file validation again.
    data_file_errors = validate_data_file(params["data_file_path"], params["method"], params["datatype"])
    errors.extend(data_file_errors)
    if errors:
        for err in errors:
            print(error(err))
        input("Execution terminated. Press ENTER to exit BBClassify.")
        return

    # Display final validated input.
    print("")
    print(note("Input validation completed successfully."))
    print("")

    # Start model parameter estimation.
    stop_loading = False
    loading_thread = threading.Thread(target=loading_animation, args=("Estimating model parameters",))
    loading_thread.start()
    import support_functions as sf
    from bbclassify import bbclassify
    output = bbclassify.bbclassify(data, params["reliability"], min_score, max_score, cut_points, true_cut_points, method, model)
    stop_loading = True
    loading_thread.join()

    # Estimate true-score moments.
    ts_raw_moments = output._tsm(
        data,
        output.max_score if method.lower() != "ll" else output.N,
        output.Parameters["lords k"]
    )
    ts_moments = [ts_raw_moments[0], ts_raw_moments[1] - ts_raw_moments[0]**2]
    ts_moments.append(
        (ts_raw_moments[2] - 3 * (ts_raw_moments[0] * ts_raw_moments[1]) + 2 * ts_raw_moments[0]**3)
        / (ts_moments[1]**0.5)**3
    )
    ts_moments.append(
        (ts_raw_moments[3] - 4 * (ts_raw_moments[0] * ts_raw_moments[2]) + 6 * (ts_raw_moments[0]**2 * ts_raw_moments[1]) - 3 * ts_raw_moments[0]**4)
        / (ts_moments[1]**0.5)**4
    )

    import statistics as stats
    import scipy.stats
    mi_reliability = (ts_moments[1]**0.5 * max_score)**2 / stats.variance(data)
    mi_sem = stats.variance(data)**0.5 * (1 - mi_reliability)**0.5

    if params["datatype"].lower() != "m":
        stop_loading = False
        loading_thread = threading.Thread(target=loading_animation, args=("Estimating model fit",))
        loading_thread.start()
        output.modelfit(minimum_expected_value=min_expected_value)
        stop_loading = True
        loading_thread.join()

        stop_loading = False
        loading_thread = threading.Thread(target=loading_animation, args=("Estimating classification accuracy",))
        loading_thread.start()
        output.accuracy()
        stop_loading = True
        loading_thread.join()
    
    # Transpose the confusion matrix for output.
    rounded_confusionmatrix = sf.add_labels(sf.array_to_strlist(output.confusionmatrix.T), "x", "t")
    
    # Compute performance metrics.
    tp = []
    tn = []
    fp = []
    fn = []
    sensitivity = []
    specificity = []
    for i in range(n_categories):
        tp_val = output.confusionmatrix[i, i]
        tp.append(tp_val)
        fp_val = output.confusionmatrix[i, :].sum() - tp_val
        fp.append(fp_val)
        fn_val = output.confusionmatrix[:, i].sum() - tp_val
        fn.append(fn_val)
        tn_val = 1 - (tp_val + fp_val + fn_val)
        tn.append(tn_val)
        if (tp_val + fn_val) == 0:
            sensitivity.append(float('nan'))
        else:
            sensitivity.append(tp_val / (tp_val + fn_val))
        if (tn_val + fp_val) == 0:
            specificity.append(float('nan'))
        else:
            specificity.append(tn_val / (tn_val + fp_val))
    
    stop_loading = False
    loading_thread = threading.Thread(target=loading_animation, args=("Estimating classification consistency",))
    loading_thread.start()
    output.consistency()
    stop_loading = True
    loading_thread.join()

    rounded_consistencymatrix = sf.add_labels(sf.array_to_strlist(output.consistencymatrix), "x", "x")
    chance_consistency = [sum(row)**2 for row in output.consistencymatrix]
    import numpy as np
    with np.errstate(divide='ignore', invalid='ignore'):
        total_chance = sum(chance_consistency)
        denom = 1 - total_chance
        if abs(denom) < 1e-10:
            coefficient_kappa = float('nan')
        else:
            coefficient_kappa = (output.Consistency - total_chance) / denom

        weighted_consistencymatrix = []
        for row in output.consistencymatrix:
            s = sum(row)
            if abs(s) < 1e-10:
                weighted_consistencymatrix.append(row)
            else:
                weighted_consistencymatrix.append(row / s)
    
        coefficient_kappas = []
        for i in range(n_categories):
            denom_i = 1 - chance_consistency[i]
            if abs(denom_i) < 1e-10:
                coefficient_kappas.append(float('nan'))
            else:
                coefficient_kappas.append((output.consistencymatrix[i][i] - chance_consistency[i]) / denom_i)
    
    if params["datatype"].lower() in ["r", "f", "c"]:
        n_observations = len(data)
    else:
        n_observations = 0
    
    category_proportions = []
    for i in range(n_categories):
        if i == 0:
            category_proportions.append(len([score for score in data if score < cut_points[0]]) / n_observations)
        elif 0 < i < (n_categories - 1):
            category_proportions.append(len([score for score in data if score >= cut_points[i - 1] and score < cut_points[i]]) / n_observations)
        else:
            category_proportions.append(len([score for score in data if score >= cut_points[-1]]) / n_observations)
    
    if params["datatype"].lower() in ["r", "f", "c"]:
        data_mean = stats.mean(data)
        data_variance = stats.variance(data)
        data_skewness = scipy.stats.skew(data)
        data_kurtosis = scipy.stats.kurtosis(data, fisher=False)
    
    # Write output to file.
    with open(input_path + "_output.txt", "w", encoding="utf-8") as file:
        file.write("******************************************************************************\n")
        file.write("***   BBClassify:  Beta-Binomial Classification Accuracy and Consistency   ***\n")
        file.write("***                              Version 1.0.0                             ***\n")
        file.write("***                                                                        ***\n")
        file.write("***                           Haakon E. Haakstad                           ***\n")
        file.write("***                                                                        ***\n")
        file.write("***            MIT License (https://opensource.org/license/mit)            ***\n")
        file.write("***                Copyright (c) 2025 Haakon Eidem Haakstad                ***\n")
        file.write("******************************************************************************\n\n")
        file.write(f"*** Listing of Input Specified in \"{file_name}\" ***\n\n")
        file.write(" Interpretation of input:\n")
        if method.lower() == "ll":
            proc_str = "Livingston and Lewis (\"LL\")."
        else:
            proc_str = "Hanson and Brennan (\"HB\")."
        file.write("  Type of Procedure:           " + proc_str + "\n")
        file.write(f"  Reliability of scores:       {params['reliability']}\n")
        file.write(f"  True-score Beta model:       {int(model)}-parameter Beta distribution\n")
        file.write(f"  Model-fit testing:           Minimum expected value of bins set to {min_expected_value}\n")
        file.write(f"  Name of data file:           {parsed_input[1][0]}\n")
        if parsed_input[1][1].lower() == "r":
            file.write("  Format of input data:        Raw scores\n")
        elif parsed_input[1][1].lower() == "f":
            file.write("  Format of input data:        Frequency distribution of raw scores\n")
            file.write(f"   - Raw-score column:         {int(parsed_input[1][2])}\n")
            file.write(f"   - Score-frequency column:   {int(parsed_input[1][3])}\n")
        else:
            file.write("  Format of input data:        " + parsed_input[1][1] + "\n")
        file.write(f"  Minimum possible score:      {min_score} " + ("(Inferred from data)" if parsed_input[1][1].lower() == "f" else "") + "\n")
        file.write(f"  Maximum possible score:      {max_score} " + ("(Inferred from data)" if parsed_input[1][1].lower() == "f" else "") + "\n")
        file.write(f"  Number of categories:        {int(parsed_input[2][0])}\n")
        file.write("  Obs.-score cut-point(s):     " + ", ".join([str(cp) for cp in cut_points]) + "\n")
        file.write("  True-score cut-point(s):     " + ", ".join([str(tc) for tc in true_cut_points]) + "\n\n")
        file.write(f"*** Summary Statistics of Data in {parsed_input[1][0]} ***\n\n")
        if params["datatype"].lower() in ["r", "f", "c"]:
            file.write(f" Number of observations:      {n_observations}\n\n")
        else:
            file.write(f" Number of observations:      {data[0]}\n\n")
        file.write(" Observed-score distribution moments:\n")
        file.write(f"  Mean:                       {sf.float_to_str(data_mean)}\n")
        file.write(f"  Variance:                   {sf.float_to_str(data_variance)} (SD = {sf.float_to_str(data_variance**0.5)})\n")
        file.write(f"  Skewness:                   {sf.float_to_str(data_skewness)}\n")
        file.write(f"  Kurtosis:                   {sf.float_to_str(data_kurtosis)}\n\n")
        file.write(" Observed category proportions:\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:                 {sf.float_to_str(category_proportions[i])}\n")
        file.write("\n*** Model Parameter Estimates ***\n\n")
        if output.failsafe_engaged:
            file.write(" WARNING: Four-parameter true-score distribution fitting procedure produced\n")
            file.write("  impermissible location parameter estimates. Reverted to a " +
                       ("two-parameter" if output.model == 2 else "three-parameter") + "\n")
            file.write("  fitting procedure with " +
                       (f'location parameter "l" set to {output.Parameters["l"]} and ' if output.model == 2 else "") +
                       f'"u" set to {output.Parameters["u"]}.\n\n')
        file.write(" Proportional true-score distribution moments:\n")
        file.write(f"  Mean:                       {sf.float_to_str(ts_moments[0])}\n")
        file.write(f"  Variance:                   {sf.float_to_str(ts_moments[1])} (SD = {sf.float_to_str(ts_moments[1]**0.5)})\n")
        file.write(f"  Skewness:                   {sf.float_to_str(ts_moments[2])}\n")
        file.write(f"  Kurtosis:                   {sf.float_to_str(ts_moments[3])}\n")
        if params["datatype"].lower() in ["r", "f", "c"]:
            file.write("\n Model implied precision:\n")
            file.write(f"  Reliability:                {sf.float_to_str(mi_reliability)}\n")
            file.write(f"  SEM:                        {sf.float_to_str(mi_sem)}\n")
        file.write(f"\n Number of moments fit:       {int(output.model)} " +
                   "(" + ", ".join(["Mean", "Variance", "Skewness", "Kurtosis"][:int(output.model)]) + ")\n\n")
        file.write(" Beta true-score distribution:\n")
        file.write(f"  Alpha:                      {sf.float_to_str(output.Parameters['alpha'])}\n")
        file.write(f"  Beta:                       {sf.float_to_str(output.Parameters['beta'])}\n")
        file.write(f"  l:                          {sf.float_to_str(output.Parameters['l'])}\n")
        file.write(f"  u:                          {sf.float_to_str(output.Parameters['u'])}\n\n")
        file.write(" Binomial error distribution:\n")
        file.write(f"  Lord's k:                   {output.Parameters['lords k']} " +
                   ("(Compound-Binomial error model)" if output.Parameters["lords k"] != 0 else "(Binomial error model)") + "\n")
        file.write(f"  Number of 'trials':         {output.N} " +
                   ("(Effective Test Length)" if method.lower() == "ll" else "(Actual Test Length)") + "\n\n")
        if params["datatype"].lower() != "m":
            file.write("*** Model Fit ***\n")
            file.write(f" Pearson's χ²:                {sf.float_to_str(output.Modelfit_chi_squared)}\n")
            file.write(f" DF:                          {int(output.Modelfit_degrees_of_freedom)}\n")
            file.write(f" p-value:                     {sf.float_to_str(output.Modelfit_p_value)}\n\n")
        file.write("*** Classification Accuracy Estimates ***\n\n")
        file.write(" Confusion matrix (transposed):\n")
        for row in rounded_confusionmatrix:
            file.write("   ".join(row) + "\n")
        if round(output.confusionmatrix.sum(), 5) != 1:
            file.write("\n NOTE: Matrix entries do not add up to 1 due to rounding errors.\n")
            file.write("  Statistics will be computed from a normalized matrix where the entries add up to 1.\n")
        file.write("\n Overall:                  Unweighted  Weighted\n")
        file.write(f"  Accuracy:                   {sf.float_to_str(output.Accuracy)}\n")
        file.write(f"  Sensitivity:                {sf.float_to_str(np.mean(sensitivity))}   {sf.float_to_str(sum([sensitivity[i] * category_proportions[i] for i in range(n_categories)]))}\n")
        file.write(f"  Specificity:                {sf.float_to_str(np.mean(specificity))}   {sf.float_to_str(sum([specificity[i] * category_proportions[i] for i in range(n_categories)]))}\n\n")
        file.write(" Category specific:\n\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:\n")
            file.write(f"   Accuracy:                  {sf.float_to_str(tp[i] + tn[i])}\n")
            file.write(f"   True Positives:            {sf.float_to_str(tp[i])}\n")
            file.write(f"   True Negatives:            {sf.float_to_str(tn[i])}\n")
            file.write(f"   Sensitivity:               {sf.float_to_str(sensitivity[i])}\n")
            file.write(f"   Specificity:               {sf.float_to_str(specificity[i])}\n\n")
        file.write("*** Classification Consistency Estimates ***\n\n")
        file.write(" Consistency matrix:\n")
        for row in rounded_consistencymatrix:
            file.write("   ".join(row) + "\n")
        if round(output.consistencymatrix.sum(), 5) != 1:
            file.write("\n NOTE: Matrix entries do not add up to 1 due to rounding errors.\n")
            file.write("  Statistics will be computed from a normalized matrix where the entries add up to 1.\n")
        file.write("\n Overall:                  Unweighted  Weighted\n")
        file.write(f"  Consistency:                {sf.float_to_str(output.Consistency)}\n")
        file.write(f"  Chance consistency:         {sf.float_to_str(sum(chance_consistency))}\n")
        file.write(f"  Coefficient Kappa:          {sf.float_to_str(coefficient_kappa)}\n\n")
        file.write(" Category specific:\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:\n")
            file.write(f"   Consistency:               {sf.float_to_str(output.consistencymatrix[i, i])}   {sf.float_to_str(weighted_consistencymatrix[i][i])}\n")
            file.write(f"   Chance consistency:        {sf.float_to_str(sum(output.consistencymatrix[i, :])**2)}\n")
            file.write(f"   Coefficient Kappa:         {sf.float_to_str(coefficient_kappas[i])}\n\n")
        file.write("\n")
        file.write(f"Analysis completed successfully. Results have been saved to the file \"{input_path + '_output.txt'}\".\n\n")
        file.write("Press ENTER to close the program...")
    
    print("\n")
    print(f"Analysis completed successfully. Results have been saved to the file \"{input_path + '_output.txt'}\".")
    print("\n")
    input("Press ENTER to close the program...")

if __name__ == "__main__":
    main()
