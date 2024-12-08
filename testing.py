import numpy as np
import pandas as pd
import math
import statistics as stats
from support_functions.betafunctions import cronbachs_alpha, rbeta4p

#np.random.seed(1234)
#N_resp, N_items, alpha, beta, l, u = 250, 10, 6, 4, .15, .85
#p_success = rbeta4p(N_resp, alpha, beta, l, u)
#arr = [np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)]

arr = { "v1": [2.566, 1.560, 1.487, 1.195, 1.425],
        "v2": [1.560, 2.493, 1.283, 0.845, 1.313],
        "v3": [1.487, 1.283, 2.462, 1.127, 1.313],
        "v4": [1.195, 0.845, 1.127, 2.769, 1.323],
        "v5": [1.425, 1.313, 1.313, 1.323, 3.356]}

arr = pd.DataFrame(arr)
arr = arr.to_numpy()
for i in range(arr.shape[0]):
    for j in range(arr.shape[0]):
        arr[i, j] = round(arr[i, j], 3)
        #arr[i, j] = arr[i, j]

#print(arr)
varlist = np.diag(arr)#[float(arr[i, i]) for i in range(arr.shape[0])]
#print(varlist)
covariance_list = [[float(arr[i + j + 1, i]) for j in range(len(arr[i:, i]) - 1)] for i in range(len(arr[0]) - 1)]
#print(covariance_list)
#for i in covariance_list: print(i)
#exit()
factor_loadings = []
for _ in range(len(varlist)):
    factor_loading = []
    covariance_list = [[float(arr[i + j + 1, i]) for j in range(len(arr[i:, i]) - 1)] for i in range(len(arr[0]) - 1)]
    for i in range(len(covariance_list[0]) - 1):
        for j in range(len(covariance_list[i + 1])):
            # If a covariance is exactly 0, consider it a rounding error and add 0.0001.
            if abs(covariance_list[i + 1][j]) == 0: covariance_list[i + 1][j] += .00001
            value = [(covariance_list[0][i] * covariance_list[0][i + j + 1])  / abs(covariance_list[i + 1][j]), 1]
            if value[0] < 0:
                value = [abs(value[0]), -1]
            factor_loading.append(value[0]**.5 * value[1])
            #print(covariance_list[0][i], covariance_list[0][i + j + 1], covariance_list[j + 1][i], round(value[0]**.5, 3))
    factor_loadings.append(stats.mean(factor_loading))
    arr = np.vstack([arr, arr[[0], :]])
    arr = np.hstack([arr, arr[:, [0]]])
    arr = arr[1:, 1:]

print(factor_loadings)


#exit()
squared_factor_loadings = [i**2 for i in factor_loadings]
factor_loadings_squared = sum(factor_loadings)**2
Omega = factor_loadings_squared / (sum([varlist[i] - squared_factor_loadings[i] for i in range(len(varlist))]) + factor_loadings_squared)
print(Omega)



#print(stats.mean(lst))

#[[ 0.237  0.023  0.006 -0.004  0.033 -0.005 -0.004 -0.011 -0.002 -0.   ]
# [ 0.023  0.249  0.016 -0.007  0.012  0.032  0.014  0.01   0.012  0.041]
# [ 0.006  0.016  0.249  0.006 -0.     0.016  0.022 -0.006  0.024  0.005]
# [-0.004 -0.007  0.006  0.248 -0.003  0.006 -0.016 -0.008  0.026  0.015]
# [ 0.033  0.012 -0.    -0.003  0.235  0.005  0.03  -0.006  0.012  0.005]
# [-0.005  0.032  0.016  0.006  0.005  0.249  0.03   0.014  0.004  0.022]
# [-0.004  0.014  0.022 -0.016  0.03   0.03   0.251  0.004  0.016  0.022]
# [-0.011  0.01  -0.006 -0.008 -0.006  0.014  0.004  0.249  0.03   0.015]
# [-0.002  0.012  0.024  0.026  0.012  0.004  0.016  0.03   0.251  0.036]
# [-0.     0.041  0.005  0.015  0.005  0.022  0.022  0.015  0.036  0.24 ]]