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
    parsed_input = [i.replace("\n", "").split(" ") for i in parsed_input]
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

    if len(parsed_input[0]) > 1:
        if not isinstance(parsed_input[0][1], (float, int)) or (parsed_input[0][1] != -1 and 0 > parsed_input[0][1] > 1):
            errors.append(f"The second value of the first line must be a value between 0 and 1. Current value is {error(parsed_input[0][1])}")

    if len(parsed_input[0]) > 2:
        if parsed_input[0][2] not in [2, 4]:
            errors.append(f"The third value of the first line must be either 2 or 4. Current value is {error(parsed_input[0][2])}.")
    
    if len(parsed_input[0]) > 3 and not isinstance(parsed_input[0][3], int):
        warnings.append(f"The fourth value of the first line must be an integer >= 0. Current value is {warning(parsed_input[0][3])}. Defaulting to 0.")
        parsed_input[0][3] = 0

    ## Line 2:
    if len(parsed_input[1]) < 2:
        errors.append("Invalid input format. Second line must contain at least 2 values.")

    if len(parsed_input[1]) > 0:

        if not os.path.isabs(parsed_input[1][0]):
            file_name = parsed_input[1][0]
            base_path = os.path.dirname(sys.executable)
            data_path = os.path.join(base_path, file_name)
        else:
            data_path = parsed_input[1][0]
            base_path = os.path.dirname(data_path)
            file_name = os.path.basename(data_path)

        if not os.path.exists(data_path):
            errors.append(f"Data file \"{file_name}\" not found at \"{base_path}\".")

    if len(parsed_input[1]) > 1:

        if parsed_input[1][1].lower() not in ["r", "f", "m", "c"]:
            errors.append(f"The second value of the second line  are \"r\" for raw data, \"f\" for frequency data, \"m\" for moment data, and \"c\" for complete data.")
        
        if parsed_input[1][1].lower() == "r":
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
        
        if parsed_input[1][1].lower() == "f":
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

        if parsed_input[1][1].lower() == "c":
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
            if len(parsed_input[2]) != parsed_input[2][0] and len(parsed_input[2]) != parsed_input[2][0]*2 - 1:
                errors.append("Number of cut-points on the third line do not match the number of specified categories.")
            if len(parsed_input[2]) == parsed_input[2][0]:
                if not all(isinstance(i, (float, int)) for i in parsed_input[2][1:]):
                    errors.append("All values on the third line must be numeric.")
            if len(parsed_input[2]) == parsed_input[2][0]*2 - 1:
                if not all(isinstance(i, float) for i in parsed_input[2][parsed_input[2][0]:]) or not all(0 < i < 1 for i in parsed_input[2][parsed_input[2][0]:]):
                    errors.append("All true-score cut-scores specified on line three must be floating-point values between 0 and 1.")

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

    data: list[list[str]] = [data[i].split(" ") for i in range(len(data))]
    data = [[j for j in i if len(j) > 0] for i in data]
    data = [[j.replace("\n", "") for j in i] for i in data]
    data: list[list[str | float]] = [[float(j) if "." in j and j.count(".") == 1 and j.replace(".", "").isnumeric() else j for j in i] for i in data]
    data: list[list[str | float | int]] = [[int(j) if isinstance(j, str) and j.isnumeric() else j for j in i] for i in data]

    if not all(isinstance(i, (float, int)) for i in [j for k in data for j in k]):
        success = False
        stop_loading = True
        thread.join()
        stop_loading = False
        print("")
        print(f" Input validation completed with {len(errors)} {"errors" if len(errors) != 1 else "error"}, {len(warnings)} {"warnings" if len(warnings) != 1 else "warning"}, and {len(notes)} {"notes" if len(notes) != 1 else "note"}.")
        print("")
        errors.append("Not all entries in the data could be interpreted as numeric. Make sure that the data file only contains numeric entries (e.g., no row or column names, decimals marked by \".\").")
        print(f"  {error("ERRORS:")}")
        for i in errors: print("   - " + i)
        print("")
        print(error("Execution terminated due to invalid input."))
        print("")
        input("Press ENTER to close BBClassify...")
        return
    
    import numpy as np, pandas as pd
    
    if parsed_input[1][1].lower() in ["r", "m"]:
        data: list[float | int] = [i for j in data for i in j]

    if parsed_input[1][1].lower() == "f":
        xcol: int = int(parsed_input[1][2] - 1)
        fcol: int = int(parsed_input[1][3] - 1)
        data: list[list[float | int]] = [[i[xcol] for _ in range(i[fcol])] for i in data]
        data: list[float | int] = [i for j in data for i in j]

    if parsed_input[1][1].lower() == "c":
        if parsed_input[0][1] == -1:
            covariance_matrix = pd.DataFrame(data).cov()
            n = covariance_matrix.shape[0]
            reliability = (n / (n - 1)) * (1 - (sum(np.diag(covariance_matrix)) / sum(sum(covariance_matrix))))
        data = [sum(i) for i in data]
    
    if len(errors) > 0:
        success = False
    stop_loading = True
    thread.join()
    stop_loading = False

    input("Press ENTER to close BBClassify...")

if __name__ == "__main__":
    main()