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