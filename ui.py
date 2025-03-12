print("WELCOME TO BBClassify: BETA-BINOMIAL CLASSIFICATION ACCURACY AND CONSISTENCY")

def main():    
    import time, threading
    
    def loading_animation(text: str):
        while not stop_loading:
            for i in range(1, 4):
                if stop_loading:
                    break
                print(f"{text + "." * i}\033[0m", end="\r")
                time.sleep(.25)
            if not stop_loading:
                print(f"{text}   ", end="\r")
        if success:
            print(f"{text}... \033[92m✓\033[0m")
        else:
            print(f"{text}... \033[91m✗\033[0m")
    
    success: bool = True    
    stop_loading: bool = False

    import os, sys

    def merge_quoted_entries(lst: list[str]):
        merged_list: list = []
        temp: list = []
        
        for entry in lst:
            if entry.startswith('"') and not entry.endswith('"'):
                temp.append(entry)
            elif temp:
                temp.append(entry)
                if entry.endswith('"'):
                    merged_list.append(" ".join(temp))
                    temp = []
            else:
                merged_list.append(entry)
        
        # Handle case where an opening quote is never closed
        if temp:
            merged_list.append(" ".join(temp))
        
        return merged_list
    
    def error(x):
        return f"\033[91m{x}\033[0m"
    def warning(x):
        return f"\033[38;5;214m{x}\033[0m"
    def note(x):
        return f"\033[92m{x}\033[0m"

    print("")
    input_path: str = input("Enter path to- or name of input file: ")

    errors, warnings, notes = [], [], []
    
    if not os.path.isabs(input_path):
        file_name = input_path
        base_path = os.path.dirname(sys.executable)
        input_path = os.path.join(base_path, input_path)

    if not os.path.exists(input_path):
        print("")
        print(f"\033[91mERROR\033[0m: File \"{file_name}\" not found at \"{base_path}\".")
        print("")
        input(f"Execution terminated. Press ENTER to exit BBClassify.")
        return
    
    with open(input_path, "r") as file:
        parsed_input: list[list[str | float | int]] = file.readlines()
    print("")
    print(f"CONTENTS OF INPUT FILE \"{file_name}\":")
    print("")
    for i in parsed_input: print(" " + i.replace("\n", ""))
    print("")
    thread = threading.Thread(target = loading_animation, args = ("Validating Input",))
    thread.start()
    parsed_input = [i.replace("\r", "").replace("\n", "").replace("\t", "").replace("\f", "").replace("\v", "").split(" ") for i in parsed_input]
    parsed_input = [merge_quoted_entries(i) for i in parsed_input]
    parsed_input = [[j for j in i if len(j) > 0] for i in parsed_input]
    parsed_input = [[float(j) if "." in j and j.count(".") == 1 and j.replace(".", "").isnumeric() else j for j in i] for i in parsed_input]
    parsed_input = [[int(j) if isinstance(j, str) and j.isnumeric() else j for j in i] for i in parsed_input]
    parsed_input = [[int(j) if not isinstance(j, str) and j % 1 == 0 else j for j in i] for i in parsed_input]
    parsed_input[1][0] = parsed_input[1][0].removeprefix("\"").removesuffix("\"")
    
    ### INPUT VALIDATION ###
    ## Overall:
    if len(parsed_input) < 3:
        errors.append("Input file must contain at least 3 lines.")
    
    ## Line 1:
    if len(parsed_input[0]) < 3:
        errors.append("First line must contain at least 3 values.")

    if len(parsed_input[0]) == 3:
        parsed_input[0].append(0)
        notes.append(f"Fourth value of the first line not specified. Defaulting to {note("0")}.")

    if len(parsed_input[0]) > 0:
        if parsed_input[0][0] not in ["hb", "HB", "hB", "Hb", "ll", "LL", "lL", "Ll"]:
            errors.append(f"The first value of the first line must be either \"hb\" or \"ll\". Current value is {error(parsed_input[0][0])}")
        method: str = parsed_input[0][0]

    if len(parsed_input[0]) > 1:
        if not isinstance(parsed_input[0][1], (float, int)) or (parsed_input[0][1] != -1 and 0 > parsed_input[0][1] > 1):
            errors.append(f"The second value of the first line must be a value between 0 and 1. Current value is {error(parsed_input[0][1])}")
        reliability: float | int = parsed_input[0][1]

    if len(parsed_input[0]) > 2:
        if parsed_input[0][2] not in [2, 4]:
            errors.append(f"The third value of the first line must be either 2 or 4. Current value is {error(parsed_input[0][2])}.")
        model: int = parsed_input[0][2]
    
    if len(parsed_input[0]) > 3:
        if not isinstance(parsed_input[0][3], int) or not parsed_input[0][3] < 0:
            warnings.append(f"The fourth value of the first line representing the minimum expected value for model fit testing must be an integer >= 0. Current value is {warning(parsed_input[0][3])}. Defaulting to 0.")
            parsed_input[0][3] = 0
        min_expected_value: int = parsed_input[0][3]
    
    ## Line 2:
    if len(parsed_input[1]) < 2:
        errors.append("Invalid input format. Second line must contain at least 2 values.")

    if len(parsed_input[1]) > 0:

        if not os.path.isabs(parsed_input[1][0]):
            file_name: str = parsed_input[1][0]
            base_path: str = os.path.dirname(sys.executable)
            data_path: str = os.path.join(base_path, file_name)
        else:
            data_path: str = parsed_input[1][0]
            base_path: str = os.path.dirname(data_path)
            file_name: str = os.path.basename(data_path)

        if not os.path.exists(data_path):
            errors.append(f"Data file \"{file_name}\" not found at \"{base_path}\".")
        
        filename: str = base_path + file_name

    if len(parsed_input[1]) > 1:

        if parsed_input[1][1].lower() not in ["r", "f", "m", "c"]:
            errors.append(f"The second value of the second line  are \"r\" for raw data, \"f\" for frequency data, \"m\" for moment data, and \"c\" for complete data.")
        
        datatype: str = parsed_input[1][1]
        
        if parsed_input[1][1] in ["r", "R"]:
            if len(parsed_input[1]) < 4:
                errors.append("The second line must contain at least 4 values when the specified data-type is \"r\".")        
            if not isinstance(parsed_input[1][2], int):
                errors.append("The third value of the second line must be an integer when the specified data-type is \"r\".")            
            if parsed_input[0][0].lower() == "hb":
                if not isinstance(parsed_input[1][3], int):
                    errors.append("The fourth value of the second line must be an integer representing test length when specified procedure is \"HB\" and data-type is \"r\".")
            else:
                if not isinstance(parsed_input[1][3], (float, int)):
                    errors.append("The fourth value of the second line must be a number representing the maximum possible test score when specified procedure is \"LL\" and data-type is \"r\".")
                if len(parsed_input[1]) == 4:
                    parsed_input[1].append(0)
                    notes.append(f"The fifth value of the second line representing the minimum possible test score not specified. Defaulting to {note("0")}.")
                if not isinstance(parsed_input[1][4], (float, int)):
                    warnings.append(f"The fifth value of the second line must be a numeric value. Current value is {warning(parsed_input[1][4])}. Defaulting to {warnings("0")}.")
                    parsed_input[1][4] = 0
            max_score = parsed_input[1][3]
            if method.lower() == "ll":
                min_score = parsed_input[1][4]
            else:
                min_score = 0
            
        if parsed_input[1][1] in ["f", "F"]:
            if len(parsed_input[1]) < 4:
                errors.append("The second line must contain at least 4 values when the specified data-type is \"f\".")
            if len(parsed_input[1]) > 2:
                if not isinstance(parsed_input[1][2], (float, int)) or parsed_input[1][2] % 1 != 0:
                    errors.append(f"The third value of the second line must be a number representing the column of the data file containing test scores. Current input is {error(parsed_input[1][2])}")
            if len(parsed_input[1]) > 3:
                if not isinstance(parsed_input[1][3], (float, int)) or parsed_input[1][3] % 1 != 0:
                    errors.append(f"The fourth value of the second line must be a number representing the column of the data file containing test scores. Current input is {error(parsed_input[1][3])}")
                if parsed_input[1][2] == parsed_input[1][3]:
                    errors.append(f"The third and fourth value of the second line must point to different columns of the data file when the specified data-type is \"f\". Current input is {error(parsed_input[1][2])} and {error(parsed_input[1][3])}.")

        if parsed_input[1][1] in ["c", "C"]:
            if len(parsed_input[1]) > 2:
                if parsed_input[0][0].lower() == "hb":
                    if not isinstance(parsed_input[1][2], int):
                        errors.append("The third value of the second line must be an integer representing test length when specified procedure is \"HB\" and data-type is \"c\".")
                else:
                    if not isinstance(parsed_input[1][2], (float, int)):
                        errors.append("The third value of the second line must be a number representing the maximum possible test score when specified procedure is \"LL\" and data-type is \"c\".")
                    if len(parsed_input) < 4:
                        notes.append(f"The fourth value of the second line representing the minimum possible test score not specified. Defaulting to {note("0")}.")
                        parsed_input[1].append(0)
                    if len(parsed_input) > 3:
                        if not isinstance(parsed_input[1][3], (float, int)):
                            warnings.append(f"The fourth value of the second line must be a numeric value. Current value is {warning(parsed_input[1][3])}. Defaulting to {warnings("0")}.")
                            parsed_input[1][3] = 0
                    if parsed_input[1][2] <= parsed_input[1][3]:
                        errors.append(f"The maximum possible test score (value: {error(parsed_input[1][2])}) must be greater than the minimum possible test score (value: {error(parsed_input[1][3])})")
    
    ## Line 3:
    if len(parsed_input[2]) < 2:
        errors.append("Invalid input format. Third line must contain at least 2 values.")
    
    if len(parsed_input) > 1:
        if isinstance(parsed_input[2][0], int):
            n_categories = parsed_input[2][0]
            if len(parsed_input[2]) != parsed_input[2][0] and len(parsed_input[2]) != parsed_input[2][0]*2 - 1:
                errors.append("Number of cut-points on the third line do not match the number of specified categories.")
            if len(parsed_input[2]) == parsed_input[2][0]:
                cut_points = parsed_input[2][1:]
                if not all(isinstance(i, (float, int)) for i in parsed_input[2][1:]):
                    errors.append("All values on the third line must be numeric.")
            if len(parsed_input[2]) == parsed_input[2][0]*2 - 1:
                if not all(isinstance(i, float) for i in parsed_input[2][parsed_input[2][0]:]) or not all(0 < i < 1 for i in parsed_input[2][parsed_input[2][0]:]):
                    errors.append("All true-score cut-scores specified on line three must be floating-point values between 0 and 1.")                
                cut_points = parsed_input[2][1:parsed_input[2][0]]
                true_cut_points = parsed_input[2][parsed_input[2][0]:]
                
    if len(errors) > 0:
        success = False
    
    stop_loading = True
    thread.join()
    stop_loading = False
    print("")
    print(f" Input validation completed with {len(errors)} {"errors" if len(errors) != 1 else "error"}, {len(warnings)} {"warnings" if len(warnings) != 1 else "warning"}, and {len(notes)} {"notes" if len(notes) != 1 else "note"}.")
    print("")
    if len(errors) > 0:
        print(f"  {error("ERRORS:")}")
        for i in errors: print("   - " + i)
        print("")
    
    if len(warnings) > 0:
        print(f"  {warning("WARNINGS:")}")
        for i in warnings: print("   - " + i)
        print("")
    
    if len(notes) > 0:
        print(f"  {note("NOTES:")}")
        for i in notes: print("   - " + i)
        print("")

    if len(errors) > 0:
        print(error("Execution terminated due to invalid input."))
        print("")
        input("Press ENTER to close BBClassify...")
        return

    errors, warnings, notes = [], [], []

    thread = threading.Thread(target = loading_animation, args = ("Validating Data",))
    thread.start()

    datafile: str = parsed_input[1][0]
    if not os.path.isabs(datafile):
        base_path = os.path.dirname(sys.executable)
        datafile = os.path.join(base_path, datafile)

    with open(datafile, "r") as file:
        data: list[str] = file.readlines()

    data = [i.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("\f", " ").replace("\v", " ") for i in data]
    data: list[list[str]] = [j for j in [data[i].split(" ") for i in range(len(data))]]
    data = [[j for j in i if len(j) > 0] for i in data]
    data: list[list[str | float]] = [[float(j) if "." in j and j.count(".") == 1 and j.replace(".", "").isnumeric() else j for j in i] for i in data]
    data: list[list[str | float | int]] = [[int(j) if isinstance(j, str) and j.isnumeric() else j for j in i] for i in data]
    
    if not all(isinstance(i, (float, int)) for i in [float(j) for k in data for j in k]):
        success = False
        stop_loading = True
        thread.join()
        stop_loading = False
        errors.append("Not all entries in the data could be interpreted as numeric. Make sure that the data file only contains numeric entries (e.g., no row or column names, decimals marked by \".\").")
        print("")
        print(f" Data validation completed with {len(errors)} {"errors" if len(errors) != 1 else "error"}, {len(warnings)} {"warnings" if len(warnings) != 1 else "warning"}, and {len(notes)} {"notes" if len(notes) != 1 else "note"}.")
        print("")
        print(f"  {error("ERRORS:")}")
        for i in errors: print("   - " + i)
        print("")
        print(error("Execution terminated due to invalid input."))
        print("")
        input("Press ENTER to close BBClassify...")
        return

    import numpy as np, pandas as pd
    import support_functions as sf
    
    if parsed_input[1][1] in ["r", "R"]:
        data: list[float | int] = [i for j in data for i in j]
        if method.lower() == "ll":
            if not all(isinstance(i, (float, int)) for i in data):
                errors.append("All entries in the data must be numeric.")
            if all(isinstance(i, (float, int)) for i in data):
                if min(data) < min_score:
                    errors.append("Minimum test score observed in the data is less than the minimum possible test score specified in the input file.")
                if max(data) > max_score:
                    errors.append("Maximum test score observed in the data is greater than the maximum possible test score specified in the input file.")
        else:
            if not all(isinstance(i, (float, int)) for i in data):
                errors.append("All entries in the data must be numeric.")
            if all(isinstance(i, (float, int)) for i in data):
                if min(data) < 0:
                    errors.append("Minimum test score observed in the data is less than 0.")
                if max(data) > max_score:
                    errors.append("Maximum test score observed in the data is greater than the maximum possible test score specified in the input file.")
                if any(i % 1 != 0 for i in data):
                    errors.append("All entries in the data must be integers.")
    
    if parsed_input[1][1] in ["m", "M"]:
        data: list[float | int] = [i for j in data for i in j]
        if len(data) < 7:
            errors.append("When moments are specified as data input, the data file must contain at least 7 values.")
        if len(data) > 0:
            if not isinstance(data[0], int):
                errors.append("When moments are specified as data input, the first value in the data file must be an integer representing sample size.")
        if len(data) > 6:
            if parsed_input[0][0].lower() == "hb":
                if not isinstance(data[6], int):
                    errors.append("When moments are specified as data input and procedure is \"HB\", the seventh value in the data file must be an integer representing test length.")
            else:
                if data[6] < data[5]:
                    errors.append("When moments are specified as data input and procedure is \"LL\", the seventh value in the data file representing the maximum possible test score must be greater than the sixth value representing the minimum possible test score.")
        if len(errors) != 0: 
            success = False
        if success:
            n_observations = data[0]
            data_mean = data[1]
            data_variance = data[2]**2
            data_skewness = data[3]
            data_kurtosis = data[4]
            min_score = data[5]
            max_score = data[6]
            k: float = 0
            if method.lower() == "ll":
                etl: float = ((data_mean - min_score) * (max_score - data_mean) - (reliability * data_variance)) / (data_variance * (1 - reliability))
                data_mean = ((data_mean - min_score) / (max_score - min_score)) * etl
                data_variance = (data_variance**.5 * (data_mean / data[1]))**2
                etl_rounded: int = int(round(etl))
            else:
                if reliability > 0:
                    k: float = sf.calculate_lords_k_from_reliability(data_mean, data_variance, reliability, max_score)
            factorial_moments: list[float] = sf.factorial_from_ordinary_moments(data_mean, data_variance, data_skewness, data_kurtosis)
            true_score_moments: list[float] = sf.true_score_moments_from_factorial_moments(factorial_moments, etl if method.lower() == "ll" else max_score, k)
            data: dict = sf.parameters_from_raw_score_moments(true_score_moments, model)

    if parsed_input[1][1] in ["f", "F"]:
        xcol: int = int(parsed_input[1][2] - 1)
        fcol: int = int(parsed_input[1][3] - 1)
        max_score = max([i[xcol] for i in data])
        min_score = min([i[xcol] for i in data])
        data: list[list[float | int]] = [[i[xcol] for _ in range(i[fcol])] for i in data]
        data: list[float | int] = [i for j in data for i in j]

    if parsed_input[1][1] in ["c", "C"]:
        if parsed_input[0][1] == -1:
            covariance_matrix: pd.DataFrame = pd.DataFrame(data).cov()
            n: int = covariance_matrix.shape[0]
            reliability: float = (n / (n - 1)) * (1 - (sum(np.diag(covariance_matrix)) / sum(sum(covariance_matrix))))
        data: list[float | int] = [sum(i) for i in data]
    
    if parsed_input[1][1] in ["r", "R","f", "F", "c", "C"]:
        if parsed_input[0][0].lower() == "hb":
            if max_score < max(data):
                errors.append("The maximum possible test score specified in the input file is less than the maximum test score observed in the data file.")
        if parsed_input[0][0].lower() == "ll":
            if min_score > min(data):
                errors.append("The minimum possible test score specified in the input file is greater than the minimum test score observed in the data file.")
            if max_score < max(data):
                errors.append("The maximum possible test score specified in the input file is less than the maximum test score observed in the data file.")
    
    if len(errors) > 0:
        success = False
    
    stop_loading = True
    thread.join()
    stop_loading = False
    print("")
    print(f" Data validation completed with {len(errors)} {"errors" if len(errors) != 1 else "error"}, {len(warnings)} {"warnings" if len(warnings) != 1 else "warning"}, and {len(notes)} {"notes" if len(notes) != 1 else "note"}.")
    print("")
    if len(errors) > 0:
        print(f"  {error("ERRORS:")}")
        for i in errors: print("   - " + i)
        print("")

    if len(warnings) > 0:
        print(f"  {warning("WARNINGS:")}")
        for i in warnings: print("   - " + i)
        print("")
    
    if len(notes) > 0:
        print(f"  {note("NOTES:")}")
        for i in notes: print("   - " + i)
        print("")

    if len(errors) > 0:
        print(error("Execution terminated due to invalid input."))
        print("")
        input("Press ENTER to close BBClassify...")
        return
    thread.join()
    stop_loading = False
    
    true_cut_points = [i / max_score for i in cut_points]

    loading_thread = threading.Thread(target = loading_animation, args = ("Estimating model parameters",))
    loading_thread.start()

    from bbclassify import bbclassify
    
    output = bbclassify.bbclassify(data, reliability, min_score, max_score, cut_points, true_cut_points, method, model)
    
    stop_loading = True
    loading_thread.join()
    
    ts_raw_moments: list = output._tsm(data, output.max_score if method.lower() != "ll" else output.effective_test_length, output.Parameters["lords k"])
    ts_moments: list = [ts_raw_moments[0], ts_raw_moments[1] - ts_raw_moments[0]**2]
    ts_moments.append((ts_raw_moments[2] - 3*(ts_raw_moments[0]*ts_raw_moments[1]) + 2*ts_raw_moments[0]**3) / (ts_moments[1]**.5)**3)
    ts_moments.append((ts_raw_moments[3] - 4*(ts_raw_moments[0] * ts_raw_moments[2]) + 6*(ts_raw_moments[0]**2 * ts_raw_moments[1]) - 3*ts_raw_moments[0]**4) / (ts_moments[1]**.5)**4)
    
    if parsed_input[1][1].lower() != "m":
        stop_loading = False
        loading_thread = threading.Thread(target = loading_animation, args = ("Estimating model fit",))
        loading_thread.start()
        output.modelfit(minimum_expected_value = min_expected_value)
        stop_loading = True
        loading_thread.join()

    stop_loading = False
    loading_thread = threading.Thread(target = loading_animation, args = ("Estimating classification accuracy",))
    loading_thread.start()
    output.accuracy()
    stop_loading = True
    loading_thread.join()
    
    import numpy as np
    rounded_confusionmatrix: list[list] = sf.add_labels(sf.array_to_strlist(output.confusionmatrix), "x", "t")
    tp, tn, fp, fn, sensitivity, specificity = [], [], [], [], [], []
    for i in range(n_categories):
        tp.append(output.confusionmatrix[i, i])
        fp.append(output.confusionmatrix[i, :].sum() - tp[i])
        fn.append(output.confusionmatrix[:, i].sum() - tp[i])
        tn.append(1 - (tp[i] + fp[i] + fn[i]))
        sensitivity.append(tp[i] / (tp[i] + fn[i]))
        specificity.append(tn[i] / (tn[i] + fp[i]))

    stop_loading = False
    loading_thread = threading.Thread(target = loading_animation, args = ("Estimating classification consistency",))
    loading_thread.start()
    output.consistency()
    stop_loading = True
    loading_thread.join()

    rounded_consistencymatrix: list[list] = sf.add_labels(sf.array_to_strlist(output.consistencymatrix), "x", "x")    
    chance_consistency: list = [sum(i)**2 for i in output.consistencymatrix]
    coefficient_kappa: float = (output.Consistency - sum(chance_consistency)) / (1 - sum(chance_consistency))

    weighted_consistencymatrix: list[list] = [i / sum(i) for i in output.consistencymatrix]
    coefficient_kappas: list = [(output.consistencymatrix[i][i] - chance_consistency[i]) / (1 - chance_consistency[i]) for i in range(n_categories)]

    if parsed_input[1][1].lower() in ["r", "f", "c"]:
        n_observations: int = len(data)

    category_proportions: list = []
    for i in range(n_categories):
        if i == 0:
            category_proportions.append(len([i for i in data if i < cut_points[0]]) / n_observations)
        elif 0 < i < (n_categories - 1):
            category_proportions.append(len([j for j in data if j >= cut_points[i - 1] and j < cut_points[i]]) / n_observations)
        else:
            category_proportions.append(len([i for i in data if i >= cut_points[len(cut_points) - 1]]) / n_observations)
    
    import statistics as stats
    import scipy.stats
    if parsed_input[1][1].lower() in ["r", "f", "c"]:
        data_mean: float = stats.mean(data)
        data_variance: float = stats.variance(data)
        data_skewness: float = scipy.stats.skew(data)
        data_kurtosis: float = scipy.stats.kurtosis(data, fisher = False)        

        mi_reliability = (ts_moments[1]**.5 * max_score)**2 / data_variance
        mi_sem = data_variance**.5 * (1 - mi_reliability)**.5
    
    
    with open(input_path + "_output.txt", "w", encoding = "utf-8") as file:

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
        file.write(f"*** Listing of Input Specified in \"{file_name}\" ***\n")
        file.write("\n")
        file.write(" Interpretation of input:\n")
        file.write(f"  Type of Procedure:           {"Livingston and Lewis (\"LL\")." if method.lower() == "ll" else "Hanson and Brennan (\"HB\")\n"}")
        file.write(f"  Reliability of scores:       {reliability}\n")
        file.write(f"  True-score Beta model:       {int(model)}-parameter Beta distribution\n")
        file.write(f"  Model-fit testing:           Minimum expected value of bins set to {parsed_input[0][3]}\n")
        file.write(f"  Name of data file:           {parsed_input[1][0]}\n")
        file.write(f"  Format of input data:        {"Raw scores" if parsed_input[1][1].lower() == "r" else "Frequency distribution of raw scores"}\n")
        if parsed_input[1][1].lower() == "f":
            file.write(f"   - Raw-score column:         {int(parsed_input[1][2])}\n")
            file.write(f"   - Score-frequency column:   {int(parsed_input[1][3])}\n")
        file.write(f"  Minimum possible score:      {min_score} {"(Inferred from data)" if parsed_input[1][1].lower() == "f" else ""}\n")
        file.write(f"  Maximum possible score:      {max_score} {"(Inferred from data)" if parsed_input[1][1].lower() == "f" else "" }\n")
        file.write(f"  Number of categories:        {int(parsed_input[2][0])}\n")
        file.write(f"  Obs.-score cut-point(s):     {", ".join([str(i) for i in cut_points])}\n")
        file.write(f"  True-score cut-point(s):     {", ".join([str((i - min_score) / (max_score - min_score)) for i in cut_points] if true_cut_points is None else [str(i) for i in true_cut_points])}\n")
        file.write("\n")
        file.write("\n")
        file.write(f"*** Summary Statistics of Data in {parsed_input[1][0]} ***\n")
        file.write("\n")
        file.write(f" Number of observations:      {len(data) if parsed_input[1][1].lower() in ["r", "f", "c"] else data[0]}\n")
        file.write("\n")
        file.write(" Observed-score distribution moments:\n")
        file.write(f"  Mean:                       {sf.float_to_str(data_mean)}\n")
        file.write(f"  Variance:                   {sf.float_to_str(data_variance)} (SD = {sf.float_to_str(data_variance**.5)})\n")
        file.write(f"  Skewness:                   {sf.float_to_str(data_skewness)}\n")
        file.write(f"  Kurtosis:                   {sf.float_to_str(data_kurtosis)}\n")
        file.write("\n")
        file.write(" Observed category proportions:\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:                 {sf.float_to_str(category_proportions[i])}\n")
        file.write("\n")
        file.write("\n")
        file.write("*** Model Parameter Estimates ***\n")
        file.write("\n")
        if output.failsafe_engaged:
            file.write( " WARNING: Four-parameter true-score distribution fitting procedure produced\n")
            file.write(f"  impermissible location parameter estimates. Reverted to a {"two" if output.model == 2 else "three"}-parameter\n")
            file.write(f"  fitting procedure with location parameter{f"s \"l\" set to {output.Parameters["l"]} and " if output.model == 2 else ""} \"u\" set to {output.Parameters["u"]}.\n")
            file.write("\n")
        file.write(f" Proportional true-score distribution moments:\n")
        file.write(f"  Mean:                       {sf.float_to_str(ts_moments[0])}\n")
        file.write(f"  Variance:                   {sf.float_to_str(ts_moments[1])} (SD = {sf.float_to_str(ts_moments[1]**.5)})\n")
        file.write(f"  Skewness:                   {sf.float_to_str(ts_moments[2])}\n")
        file.write(f"  Kurtosis:                   {sf.float_to_str(ts_moments[3])}\n")
        if parsed_input[1][1].lower() in ["r", "f", "c"]:
            file.write("\n")
            file.write(" Model implied precision:\n")
            file.write(f"  Reliability:                {sf.float_to_str(mi_reliability)}\n")
            file.write(f"  SEM:                        {sf.float_to_str(mi_sem)}\n")
        file.write("\n")
        file.write(f" Number of moments fit:       {int(output.model)} ({", ".join(["Mean", "Variance", "Skewness", "Kurtosis"][:int(output.model)])})\n")
        file.write("\n")
        file.write(f" Beta true-score distribution:\n")
        file.write(f"  Alpha:                      {sf.float_to_str(output.Parameters["alpha"])}\n")
        file.write(f"  Beta:                       {sf.float_to_str(output.Parameters["beta"])}\n")
        file.write(f"  l:                          {sf.float_to_str(output.Parameters["l"])}\n")
        file.write(f"  u:                          {sf.float_to_str(output.Parameters["u"])}\n")
        file.write("\n")
        file.write(f" Binomial error distribution:\n")
        file.write(f"  Lord's k:                   {output.Parameters["lords k"]} ({"Compound-Binomial error model" if output.Parameters["lords k"] != 0 else "Binomial error model"})\n")
        file.write(f"  Number of 'trials':         {output.N} ({"Effective Test Length" if method.lower() == "ll" else "Actual Test Length"})\n")
        file.write("\n")
        file.write("\n")
        if parsed_input[1][1].lower() != "m":
            file.write("*** Model Fit ***\n")
            file.write(f" Pearson's \u03C7\u00B2:                {sf.float_to_str(output.Modelfit_chi_squared)}\n")
            file.write(f" DF:                          {int(output.Modelfit_degrees_of_freedom)}\n")
            file.write(f" p-value:                     {sf.float_to_str(output.Modelfit_p_value)}\n")
            file.write("\n")
            file.write("\n")
        file.write("*** Classification Accuracy Estimates ***\n")
        file.write("\n")
        file.write(" Confusion matrix:\n")
        for i in range(len(rounded_confusionmatrix)): file.write(f"{"   ".join(rounded_confusionmatrix[i])}\n")
        if output.confusionmatrix.round(5).sum() != 1:
            file.write( "\n NOTE: Matrix entries do not add up to 1 due to rounding errors.\n  Statistics will be computed from a normalized matrix where the\n  entries do add up to 1.\n")
        file.write("\n")
        file.write(f" Overall:                  Unweighted  Weighted\n")
        file.write(f"  Accuracy:                   {sf.float_to_str(output.Accuracy)}\n")
        file.write(f"  Sensitivity:                {sf.float_to_str(np.mean(sensitivity))}   {sf.float_to_str(sum([sensitivity[i] * category_proportions[i] for i in range(n_categories)]))}\n")
        file.write(f"  Specificity:                {sf.float_to_str(np.mean(specificity))}   {sf.float_to_str(sum([specificity[i] * category_proportions[i] for i in range(n_categories)]))}\n")
        file.write("\n")
        file.write(" Category specific:\n")
        file.write("\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:\n")
            file.write(f"   Accuracy:                  {sf.float_to_str(tp[i] + tn[i])}\n")
            file.write(f"   True Positives:            {sf.float_to_str(tp[i])}\n")
            file.write(f"   True Negatives:            {sf.float_to_str(tn[i])}\n")
            file.write(f"   Sensitivity:               {sf.float_to_str(sensitivity[i])}\n")
            file.write(f"   Specificity:               {sf.float_to_str(specificity[i])}\n")
            file.write("\n")
        file.write("\n")
        file.write("*** Classification Consistency Estimates ***\n")
        file.write("\n")
        file.write(" Consistency matrix:\n")
        for i in range(len(rounded_consistencymatrix)): file.write(f"{"   ".join(rounded_consistencymatrix[i])}\n")
        if output.consistencymatrix.round(5).sum() != 1:
            file.write( "\n NOTE: Matrix entries do not add up to 1 due to rounding errors.\n  Statistics will be computed from a normalized matrix where the\n  entries do add up to 1.\n")
        file.write("\n")
        file.write(f" Overall:                  Unweighted  Weighted\n") 
        file.write(f"  Consistency:                {sf.float_to_str(output.Consistency)}\n")
        file.write(f"  Chance consistency:         {sf.float_to_str(sum(chance_consistency))}\n")
        file.write(f"  Coefficient Kappa:          {sf.float_to_str(coefficient_kappa)}\n")
        file.write("\n")
        file.write(" Category specific:\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:\n")
            file.write(f"   Consistency:               {sf.float_to_str(output.consistencymatrix[i, i])}   {sf.float_to_str(weighted_consistencymatrix[i][i])}\n")
            file.write(f"   Chance consistency:        {sf.float_to_str(sum(output.consistencymatrix[i, :])**2)}\n")
            file.write(f"   Coefficient Kappa:         {sf.float_to_str(coefficient_kappas[i])}\n")
            file.write("\n")
    print("\n")
    print(f"Analysis completed successfully. Results have been saved to the file \"{input_path + "_output.txt"}\".")
    print("\n")
    input("Press ENTER to close the program...")
    
if __name__ == "__main__":
    main()