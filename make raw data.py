

with open("act288m", "r") as file:
    data = file.read()
    data = data.split("\n")
    data = data[0:-1]

data = [i.replace("\r", " ").replace("\n", " ").replace("\t", " ").replace("\f", " ").replace("\v", " ") for i in data]
data: list[list[str]] = [j for j in [data[i].split(" ") for i in range(len(data))]]
data = [[j for j in i if len(j) > 0] for i in data]
data: list[list[str | float]] = [[float(j) if "." in j and j.count(".") == 1 and j.replace(".", "").isnumeric() else j for j in i] for i in data]
data: list[list[str | float | int]] = [[int(j) if isinstance(j, str) and j.isnumeric() else j for j in i] for i in data]

xcol: int = 0
fcol: int = 1
data: list[list[float | int]] = [[i[xcol] for _ in range(i[fcol])] for i in data]
#data: list[float | int] = [i for j in data for i in j]

with open("act288m_raw", "w") as file:
    for i in data:
        for j in i:
            file.write(str(j) + " ")