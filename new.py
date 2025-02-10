print("WELCOME TO BBClassify: BETA-BINOMIAL CLASSIFICATION ACCURACY AND CONSISTENCY")

def main():    
    import time
    import threading
    
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
    print(" ")
    input_path: str = input("Enter path to- or name of input file: ")

    errors, warnings, notes = [], [], []
    
    if not os.path.isabs(input_path):
        file_name = input_path
        print(file_name)
        base_path = os.path.dirname(sys.executable)
        print(base_path)
        input_path = os.path.join(base_path, input_path)
        print(input_path)

    if not os.path.exists(input_path):
        print(f"ERROR: File \"{file_name}\" not found at \"{base_path}\".")
        input(f"Execution terminated. Press enter to exit program.")
        return
    
    with open(input_path, "r") as file:
        parsed_input: list[list[str | float | int]] = file.readlines()
    
    parsed_input = [i.replace("\n", "").split(" ") for i in parsed_input]
    parsed_input = [merge_quoted_entries(i) for i in parsed_input]
    parsed_input = [[j for j in i if len(j) > 0] for i in parsed_input]
    parsed_input = [[float(j) if "." in j and j.count(".") == 1 and j.replace(".", "").isnumeric() else j for j in i] for i in parsed_input]
    parsed_input = [[int(j) if isinstance(j, str) and j.isnumeric() else j for j in i] for i in parsed_input]
    parsed_input = [[int(j) if not isinstance(j, str) and j % 1 == 0 else j for j in i] for i in parsed_input]
    
    ### INPUT VALIDATION ###
    ## Overall:
    if len(parsed_input) < 3:
        errors.append("Invalid input format. Input file must contain at least 3 lines.")
    
    ## Line 1:
    if len(parsed_input[0]) < 3:
        errors.append("Invalid input format. First line must contain at least 3 values.")
    if len(parsed_input[0]) == 3:
        parsed_input[0].append(0)
        notes.append("Fourth value of the first line not specified. Defaulting to 0.")
    if parsed_input[0][0] not in ["hb", "HB", "hB", "Hb", "ll", "LL", "lL", "Ll"]:
        errors.append("Invalid input format. First value of the first line must be either \"hb\" or \"ll\".")
    if not isinstance(parsed_input[0][1], (float, int)) or 0 > parsed_input[0][1] > 1:
        errors.append("Invalid input format. Second value of the first line must be a value between 0 and 1.")
    if parsed_input[0][2] not in [2, 4]:
        errors.append("Invalid input format. Third value of the first line must be either 2 or 4.")
    if len(parsed_input[0]) >= 4 and not isinstance(parsed_input[0][3], int):
        parsed_input[0][3] = 0
        warnings.append("Invalid input format. Fourth value of the first line must be an integer. Defaulting to 0.")

    ## Line 2:
    if len(parsed_input[1]) < 2:
        errors.append("Invalid input format. Second line must contain at least 2 values.")
    
    if not os.path.isabs(parsed_input[1][0]):
        file_name = parsed_input[1][0]
        base_path = os.path.dirname(sys.executable)
        data_path = os.path.join(base_path, data_path)
    else:
        data_path = parsed_input[1][0]
        base_path = os.path.dirname(data_path)
        file_name = os.path.basename(data_path)
    if not os.path.exists(data_path):
        errors.append(f"Data file \"{file_name}\" not found at \"{base_path}\".")
    
    if parsed_input[1][1].lower() not in ["r", "f", "m", "c"]:
        errors.append(f"Invalid input type ({parsed_input[1][1]}). Valid input types are \"r\" for raw data, \"f\" for frequency data, \"m\" for moment data, and \"c\" for complete data.")
        print("BBClassify encountered the following errors while parsing the input:\n")
        for i in errors: print(i)
        input("Execution terminated. Press ENTER to close the program...")
        return
    
    if parsed_input[1][1].lower() == "r":
        if len(parsed_input[1]) < 4:
            errors.append("Invalid input format. Second line must contain at least 4 values when the specified data-type is \"r\".")
            print("BBClassify encountered the following errors while parsing the input:\n")
            for i in errors: print(i)
            input("Execution terminated. Press ENTER to close the program...")
            return
    
        if not isinstance(parsed_input[1][2], int):
            errors.append("Invalid input format. Third value of the second line must be an integer when the specified data-type is \"r\".")
            print("BBClassify encountered the following errors while parsing the input:\n")
            for i in errors: print(i)
            input("Execution terminated. Press ENTER to close the program...")
            return
        
        if parsed_input[0][0].lower() == "hb":
            if not isinstance(parsed_input[1][3], int):
                errors.append("Invalid input format. Fourth value of the second line must be an integer representing test length when specified procedure is \"HB\" and data-type is \"r\".")
                print("BBClassify encountered the following errors while parsing the input:\n")
                for i in errors: print(i)
                input("Execution terminated. Press ENTER to close the program...")
                return
        else:
            if not isinstance(parsed_input[1][3], (float, int)):
                errors.append("Invalid input format. Fourth value of the second line must be a number representing the maximum possible test score when specified procedure is \"LL\" and data-type is \"r\".")
                print("BBClassify encountered the following errors while parsing the input:\n")
                for i in errors: print(i)
                input("Execution terminated. Press ENTER to close the program...")
                return
                
            if len(parsed_input[1]) == 4:
                parsed_input[1].append(0)
                notes.append("Fifth value of the second line not specified. Defaulting to 0.")
            if not isinstance(parsed_input[1][4], (float, int)):
                errors.append("Invalid input format. Fifth value of the second line must be a number representing minimum possible test scpre when specified procedure is \"LL\" and data-type is \"r\".")
    
    for i in parsed_input: print(i)
    input("Press enter to exit.")

if __name__ == "__main__":
    main()