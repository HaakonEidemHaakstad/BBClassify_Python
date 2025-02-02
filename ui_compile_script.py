
print("Welcome to BBClassify: Beta-Binomial Classification Accuracy and Consistency\n".upper())

import threading
import time
success = True
def loading_animation(text: str = "Initializing program"):
    while not stop_loading:
        for i in range(1, 4):
            if stop_loading:
                break
            print(f"{text + '\033[38;5;214m.' * i}\033[0m", end="\r")
            time.sleep(.25)
        if not stop_loading:
            print(f"{text}   ", end="\r")
    print(f"{text}... \033[92m✓\033[0m        \n")

stop_loading = False
loading_thread = threading.Thread(target = loading_animation)
loading_thread.start()
from support_functions import read_and_parse_input, read_and_parse_data, float_to_str, array_to_strlist, add_labels
stop_loading = True
loading_thread.join()

def main():
    success = True
    input_file: str = input("Enter path to- or name of the input file: ")

    try:
        input_file_raw: list[str] = read_and_parse_input(input_file, True, True)
    except:
        print("Error reading input file.")
        input("Execution terminated. Press ENTER to close the program...")
        return    
   
    print("")
    print("CONTENTS OF INPUT FILE:")
    for i in input_file_raw:
        print("  " + i, end = "")
    print("")
    def loading_animation(text: str = "Loading libraries"):
        while not stop_loading:
            for i in range(1, 4):
                if stop_loading:
                    break
                print(f"{text + '\033[38;5;214m.' * i}\033[0m", end="\r")
                time.sleep(.25)
            if not stop_loading:
                print(f"{text}   ", end="\r")
        if success:
            print(f"{text}... \033[92m✓\033[0m")
        else:
            print(f"{text}... \033[91m✗\033[0m")
    
    stop_loading = False
    loading_thread = threading.Thread(target = loading_animation, args = ("Parsing input",))
    loading_thread.start()
    
    input_file_name: str = input_file[::-1].split("/")[0][::-1]
   
    try:
        input_file: list = read_and_parse_input(input_file, False, True)
    except:
        print(f"Error parsing input file \"{input_file}\". Check whether the file is correctly formatted.")
        print("")
        input("Execution terminated. Press ENTER to close the program...")
        return
    
    method: str = input_file[0][0]
    model: int = input_file[0][2]
    minimum_expected_value: float = input_file[0][3]

    try:
        data: list = read_and_parse_data(input_file, True)
    except:
        success = False
        stop_loading = True
        loading_thread.join()
        print("Error reading data file.\n")
        input("Execution terminated. Press ENTER to close the program...")
        return
    
    n_observations: int = len(data[0])
    n_categories: int = int(input_file[2][0])
    moments: list = ["Mean", "Variance", "Skewness", "Kurtosis"]
    import statistics as stats
    import scipy.stats
    mean: float = stats.mean(data[0])
    variance: float = stats.variance(data[0])
    skewness: float = scipy.stats.skew(data[0])
    kurtosis: float = scipy.stats.kurtosis(data[0], fisher = False)
    reliability: float = input_file[0][1]
    if input_file[1][1].lower() == "r":
        min_score: float = input_file[1][4] if input_file[0][0].lower() == "ll" else 0
        max_score: float = input_file[1][3]
    if input_file[1][1].lower() == "f":
        min_score: float = data[1][0][0]
        max_score: float = data[1][len(data[1]) - 1][0]
    cut_scores = [float(i) if float(i) % 1 != 0 else int(i) for i in input_file[2][1]]
    if len(input_file[2]) == 3:
        cut_truescores: list = input_file[2][2]
    else:
        cut_truescores = None    
    
    stop_loading = True
    loading_thread.join()

    print(" Line 1:")
    print(f"  Type of Procedure:           {"Livingston and Lewis (\"LL\")." if method.lower() == "ll" else "Hanson and Brennan (\"HB\")"}")
    print(f"  Reliability of scores:       {reliability}")
    print(f"  True-score Beta model:       {int(model)}-parameter Beta distribution")
    print(f"  Model-fit testing:           Minimum expected value of bins set to {minimum_expected_value}")
    print(" Line 2:")
    print(f"  Name of data file:           {input_file[1][0]}")
    print(f"  Format of input data:        {"Raw scores" if input_file[1][1].lower() == "r" else "Frequency distribution of raw scores"}")
    if input_file[1][1].lower() == "f":
        print(f"   - Raw-score column:         {int(input_file[1][2])}")
        print(f"   - Score-frequency column:   {int(input_file[1][3])}")
    print(f"  Minimum possible score:      {min_score} {"(Inferred from data)" if input_file[1][1].lower() == "f" else ""}")
    print(f"  Maximum possible score:      {max_score} {"(Inferred from data)" if input_file[1][1].lower() == "f" else "" }")
    print(" Line 3:")
    print(f"  Number of categories:        {int(input_file[2][0])}")
    print(f"  Obs.-score cut-point(s):     {", ".join([str(i) for i in cut_scores])}")
    print(f"  True-score cut-point(s):     {", ".join([str((i - min_score) / (max_score - min_score)) for i in cut_scores] if cut_truescores is None else [str(i) for i in cut_truescores])}")
    
    print("\n")
    print("PERFORMING ANALYSIS:")
    
    stop_loading = False
    loading_thread = threading.Thread(target = loading_animation, args = (" Estimating model parameters",))
    loading_thread.start()    
    from bbclassify import bbclassify
    output = bbclassify.bbclassify(data = data[0],
                                   reliability = reliability,
                                   min_score = min_score,
                                   max_score = max_score,
                                   cut_scores = cut_scores,
                                   cut_truescores = cut_truescores,
                                   method = method,
                                   model = 4,
                                   failsafe = True)
    
    stop_loading = True
    loading_thread.join()
    
    ts_raw_moments: list = output._tsm(data[0], output.max_score if method.lower() != "ll" else output.effective_test_length, output.Parameters["lords k"])
    ts_moments: list = [ts_raw_moments[0], ts_raw_moments[1] - ts_raw_moments[0]**2]
    ts_moments.append((ts_raw_moments[2] - 3*(ts_raw_moments[0]*ts_raw_moments[1]) + 2*ts_raw_moments[0]**3) / (ts_moments[1]**.5)**3)
    ts_moments.append((ts_raw_moments[3] - 4*(ts_raw_moments[0] * ts_raw_moments[2]) + 6*(ts_raw_moments[0]**2 * ts_raw_moments[1]) - 3*ts_raw_moments[0]**4) / (ts_moments[1]**.5)**4)
    if input_file[1][1].lower() != "m":
        stop_loading = False
        loading_thread = threading.Thread(target = loading_animation, args = (" Estimating model fit",))
        loading_thread.start()
        output.modelfit(minimum_expected_value = minimum_expected_value)
        stop_loading = True
        loading_thread.join()

    stop_loading = False
    loading_thread = threading.Thread(target = loading_animation, args = (" Estimating classification accuracy",))
    loading_thread.start()
    output.accuracy()
    stop_loading = True
    loading_thread.join()
    
    import numpy as np
    rounded_confusionmatrix: list[list] = add_labels(array_to_strlist(output.confusionmatrix), "x", "t")
    tp, tn, fp, fn, sensitivity, specificity = [], [], [], [], [], []
    for i in range(n_categories):
        tp.append(output.confusionmatrix[i, i])
        fp.append(output.confusionmatrix[i, :].sum() - tp[i])
        fn.append(output.confusionmatrix[:, i].sum() - tp[i])
        tn.append(1 - (tp[i] + fp[i] + fn[i]))
        sensitivity.append(tp[i] / (tp[i] + fn[i]))
        specificity.append(tn[i] / (tn[i] + fp[i]))

    stop_loading = False
    loading_thread = threading.Thread(target = loading_animation, args = (" Estimating classification consistency",))
    loading_thread.start()
    output.consistency()
    stop_loading = True
    loading_thread.join()

    rounded_consistencymatrix: list[list] = add_labels(array_to_strlist(output.consistencymatrix), "x", "x")    
    chance_consistency: list = [sum(i)**2 for i in output.consistencymatrix]
    coefficient_kappa: float = (output.Consistency - sum(chance_consistency)) / (1 - sum(chance_consistency))

    weighted_consistencymatrix: list[list] = [i / sum(i) for i in output.consistencymatrix]
    coefficient_kappas: list = [(output.consistencymatrix[i][i] - chance_consistency[i]) / (1 - chance_consistency[i]) for i in range(n_categories)]

    category_proportions: list = []
    for i in range(n_categories):
        if i == 0:
            category_proportions.append(len([i for i in data[0] if i < input_file[2][1][0]]) / n_observations)
        elif 0 < i < (n_categories - 1):
            category_proportions.append(len([j for j in data[0] if j >= cut_scores[i - 1] and j < cut_scores[i]]) / n_observations)
        else:
            category_proportions.append(len([i for i in data[0] if i >= cut_scores[len(cut_scores) - 1]]) / n_observations)
    
    with open(input_file_name + "_output.txt", "w", encoding = "utf-8") as file:

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
        file.write(" Raw input:\n")
        for i in input_file_raw:
            file.write("  " + i)
        file.write("\n")
        file.write(" Interpretation of input:\n")
        file.write(f"  Type of Procedure:           {"Livingston and Lewis (\"LL\")." if method.lower() == "ll" else "Hanson and Brennan (\"HB\")\n"}")
        file.write(f"  Reliability of scores:       {reliability}\n")
        file.write(f"  True-score Beta model:       {int(model)}-parameter Beta distribution\n")
        file.write(f"  Model-fit testing:           Minimum expected value of bins set to {minimum_expected_value}\n")
        file.write(f"  Name of data file:           {input_file[1][0]}\n")
        file.write(f"  Format of input data:        {"Raw scores" if input_file[1][1].lower() == "r" else "Frequency distribution of raw scores"}\n")
        if input_file[1][1].lower() == "f":
            file.write(f"   - Raw-score column:         {int(input_file[1][2])}\n")
            file.write(f"   - Score-frequency column:   {int(input_file[1][3])}\n")
        file.write(f"  Minimum possible score:      {min_score} {"(Inferred from data)" if input_file[1][1].lower() == "f" else ""}\n")
        file.write(f"  Maximum possible score:      {max_score} {"(Inferred from data)" if input_file[1][1].lower() == "f" else "" }\n")
        file.write(f"  Number of categories:        {int(input_file[2][0])}\n")
        file.write(f"  Obs.-score cut-point(s):     {", ".join([str(i) for i in cut_scores])}\n")
        file.write(f"  True-score cut-point(s):     {", ".join([str((i - min_score) / (max_score - min_score)) for i in cut_scores] if cut_truescores is None else [str(i) for i in cut_truescores])}\n")
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
            file.write( " WARNING: Four-parameter true-score distribution fitting procedure produced\n")
            file.write(f"  impermissible location parameter estimates. Reverted to a {"two" if output.model == 2 else "three"}-parameter\n")
            file.write(f"  fitting procedure with location parameter{f"s \"l\" set to {output.Parameters["l"]} and " if output.model == 2 else ""} \"u\" set to {output.Parameters["u"]}.\n")
            file.write("\n")
        file.write(f" Proportional true-score distribution moments:\n")
        file.write(f"  Mean:                       {float_to_str(ts_moments[0])}\n")
        file.write(f"  Variance:                   {float_to_str(ts_moments[1])} (SD = {float_to_str(ts_moments[1]**.5)})\n")
        file.write(f"  Skewness:                   {float_to_str(ts_moments[2])}\n")
        file.write(f"  Kurtosis:                   {float_to_str(ts_moments[3])}\n")
        file.write("\n")
        file.write(f" Reliability:                 {float_to_str((ts_moments[1]**.5 * max_score)**2 / variance)}\n")
        file.write("\n")
        file.write(f" Number of moments fit:       {int(output.model)} ({", ".join(moments[:int(output.model)])})\n")
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
        if input_file[1][1].lower() != "m":
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
        if output.confusionmatrix.round(5).sum() != 1:
            file.write( "\n NOTE: Matrix entries do not add up to 1 due to rounding errors.\n  Statistics will be computed from a normalized matrix where the\n  entries do add up to 1.\n")
        file.write("\n")
        file.write(f" Overall:                  Unweighted  Weighted\n")
        file.write(f"  Accuracy:                   {float_to_str(output.Accuracy)}\n")
        file.write(f"  Sensitivity:                {float_to_str(np.mean(sensitivity))}   {float_to_str(sum([sensitivity[i] * category_proportions[i] for i in range(n_categories)]))}\n")
        file.write(f"  Specificity:                {float_to_str(np.mean(specificity))}   {float_to_str(sum([specificity[i] * category_proportions[i] for i in range(n_categories)]))}\n")
        file.write("\n")
        file.write(" Category specific:\n")
        file.write("\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:\n")
            file.write(f"   Accuracy:                  {float_to_str(tp[i] + tn[i])}\n")
            file.write(f"   True Positives:            {float_to_str(tp[i])}\n")
            file.write(f"   True Negatives:            {float_to_str(tn[i])}\n")
            file.write(f"   Sensitivity:               {float_to_str(sensitivity[i])}\n")
            file.write(f"   Specificity:               {float_to_str(specificity[i])}\n")
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
        file.write(f"  Consistency:                {float_to_str(output.Consistency)}\n")
        file.write(f"  Chance consistency:         {float_to_str(sum(chance_consistency))}\n")
        file.write(f"  Coefficient Kappa:          {float_to_str(coefficient_kappa)}\n")
        file.write("\n")
        file.write(" Category specific:\n")
        for i in range(n_categories):
            file.write(f"  Category {i + 1}:\n")
            file.write(f"   Consistency:               {float_to_str(output.consistencymatrix[i, i])}   {float_to_str(weighted_consistencymatrix[i][i])}\n")
            file.write(f"   Chance consistency:        {float_to_str(sum(output.consistencymatrix[i, :])**2)}\n")
            file.write(f"   Coefficient Kappa:         {float_to_str(coefficient_kappas[i])}\n")
            file.write("\n")
    print("\n")
    print(f"Analysis completed successfully. Results have been saved to the file \"{input_file_name + "_output.txt"}\".")
    print("\n")
    input("Press ENTER to close the program...")
if __name__ == "__main__":
    main()