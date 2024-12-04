import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype = int)
arr = np.vstack([arr, arr[[0], :]])
arr = np.hstack([arr, arr[:, [0]]])
#arr = arr[1:, 1:]

x = []
for i in range(len(arr[0]) - 1):
    y = []
    for j in range(1 ,len(arr[(i + 1):, i] + 1)):
        y.append(float(arr[j, i]))
    x.append(y)
#x = [[float(arr[i, j]) for j in range(len(arr[(i + 1):, i]) - 1)] for i in range(len(arr[0]) - 1)]
print(arr)
print(x)