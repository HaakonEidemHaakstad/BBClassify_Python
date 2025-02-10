print("Welcome to BBCLassify: Beta Binomial Classification Accuracy and Consistency\n".upper())
import threading
import time

success = True
def loading_animation(text: str = "Initializing program"):
    while not stop_loading:
        for i in range(1, 4):
            if stop_loading:
                break
            print(f"{text + "." * i}", end="\r")
            time.sleep(.25)
        if not stop_loading:
            print(f"{text}   ", end="\r")
    print(f"{text}... \033[92m✓\033[0m\n")
stop_loading = False

loading_thread = threading.Thread(target = loading_animation)
loading_thread.start()
from support_functions import read_and_parse_input, read_and_parse_data, float_to_str, array_to_strlist, add_labels
stop_loading = True
loading_thread.join()

def merge_quoted_entries(lst: list[str]):
    merged_list = []
    temp = []
    
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

def terminal_output(type: str, message: str) -> str:
    if type == "error":
        return f"\033[91m ERROR:\033[0m {message}"
    elif type == "warn":
        return f"\033[38;5;214m WARNING:\033[0m {message}"
    elif type == "note":
        return f"\033[92m NOTE:\033[0m {message}"

def main():
    input_path = input("Enter the path to the input file: ")    
    import os
    errors, warnings, notes = [], [], []
    if not os.path.isabs(input_path):
        file_name = input_path
        base_path = os.path.dirname(os.path.abspath(__file__))
        input_path = os.path.join(base_path, input_path)

    if not os.path.exists(input_path):
        return terminal_output("error", f"Input file \"{file_name}\" not found at \"{base_path}\".")
    
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
        errors.append(terminal_output("error", "Invalid input format. Input file must contain at least 3 lines."))
    
    ## Line 1:
    if len(parsed_input[0]) < 3:
        errors.append(terminal_output("error", "Invalid input format. First line must contain at least 3 values."))
    if len(parsed_input[0]) == 3:
        parsed_input[0].append(0)
        notes.append(terminal_output("note", "Fourth value of the first line not specified. Defaulting to 0."))
    if parsed_input[0][0] not in ["hb", "HB", "hB", "Hb", "ll", "LL", "lL", "Ll"]:
        errors.append(terminal_output("error", "Invalid input format. First value of the first line must be either \"hb\" or \"ll\"."))
    if not isinstance(parsed_input[0][1], (float, int)) or 0 > parsed_input[0][1] > 1:
        errors.append(terminal_output("error", "Invalid input format. Second value of the first line must be a value between 0 and 1."))
    if parsed_input[0][2] not in [2, 4]:
        errors.append(terminal_output("error", "Invalid input format. Third value of the first line must be either 2 or 4."))
    if len(parsed_input[0]) >= 4 and not isinstance(parsed_input[0][3], int):
        parsed_input[0][3] = 0
        warnings.append(terminal_output("warn", "Invalid input format. Fourth value of the first line must be an integer. Defaulting to 0."))

    ## Line 2:
    if len(parsed_input[1]) < 2:
        errors.append(terminal_output("error", "Invalid input format. Second line must contain at least 2 values."))
    
    if not os.path.isabs(parsed_input[1][0]):
        file_name = parsed_input[1][0]
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, data_path)
    else:
        data_path = parsed_input[1][0]
        base_path = os.path.dirname(data_path)
        file_name = os.path.basename(data_path)
    if not os.path.exists(data_path):
        errors.append(terminal_output("error", f"Data file \"{file_name}\" not found at \"{base_path}\"."))
    
    if parsed_input[1][1].lower() not in ["r", "f", "m", "c"]:
        return terminal_output("error", f"Invalid input type ({parsed_input[1][1]}). Valid input types are \"r\" for raw data, \"f\" for frequency data, \"m\" for moment data, and \"c\" for complete data.")
    
    if parsed_input[1][1].lower() == "r":
        if len(parsed_input[1]) < 4:
            return terminal_output("error", "Invalid input format. Second line must contain at least 4 values when the specified data-type is \"r\".")
        if not isinstance(parsed_input[1][2], int):
            return terminal_output("error", "Invalid input format. Third value of the second line must be an integer when the specified data-type is \"r\".")
        
        if parsed_input[0][0].lower() == "hb":
            if not isinstance(parsed_input[1][3], int):
                return terminal_output("error", "Invalid input format. Fourth value of the second line must be an integer representing test length when specified procedure is \"HB\" and data-type is \"r\".")
        else:
            if not isinstance(parsed_input[1][3], (float, int)):
                return terminal_output("error", "Invalid input format. Fourth value of the second line must be a number representing the maximum possible test score when specified procedure is \"LL\" and data-type is \"r\".")
            if len(parsed_input[1]) == 4:
                parsed_input[1].append(0)
                notes.append(terminal_output("note", "Fifth value of the second line not specified. Defaulting to 0."))
            if not isinstance(parsed_input[1][4], (float, int)):
                errors.append(terminal_output("error", "Invalid input format. Fifth value of the second line must be a number representing minimum possible test scpre when specified procedure is \"LL\" and data-type is \"r\"."))

    
    print(warnings)
    
    """                                        ### DATA VALIDATION ###
    with open(data_path, "r") as file:
        data: list[str] = file.readlines()
    input_type = parsed_input[1][1]

    if input_type.lower() == "r":
        data = [i.split(" ") for i in data]
        data = [[j for j in i if len(j) > 0] for i in data]    
        data = [[j.replace("\n", "") for j in i] for i in data]
        data = [[int(j) if float(j) % 1 == 0 else float(j) for j in i] for i in data]
        data = [j for i in data for j in i]

    if input_type.lower() == "f":
        data = [i.split(" ") for i in data]
        data = [[j for j in i if len(j) > 0] for i in data]
        data = [[j.replace("\n", "") for j in i] for i in data]
        data = [[int(j) if float(j) % 1 == 0 else float(j) for j in i] for i in data]
        data = [[j[xcol - 1]]*j[fcol - 1] for j in data]
        data = [j for i in data for j in i]
        
    if input_type.lower() == "m":    
        data = [data[i].split(" ") for i in range(len(data))]
        data = [[j.replace("\n", "") for j in i] for i in data]
        data = [[j for j in i if len(j) > 0] for i in data]
        data = [[int(j) if float(j) % 1 == 0 else float(j) for j in i] for i in data]
        data = [i for j in data for i in j]

    if input_type.lower() == "c":
        data = [i.replace("\n", "").split(" ") for i in data]
        data = [[int(j) if float(j) % 1 == 0 else float(j) for j in i] for i in data]
        data = [sum(i) for i in data] 
    """
if __name__ == "__main__":
    main()