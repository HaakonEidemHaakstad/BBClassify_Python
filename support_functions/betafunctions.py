import scipy.special
from scipy.integrate import quad
from scipy.stats import binom
import statistics as stats
import math
import pandas as pd
import numpy as np
import csv

def string_to_number(s: str, notlist: bool = True):
    """
    Turn a string with numbers delimited by commas into numbers.

    Parameters
    ----------
    s: str
        A string of numbers separated by commas.
    notlist: bool
        Return a single value if the list is of length 1.
    
    Returns
    -------
    list
        - If 's' contains more than one number and notlist is False.
    float
        - If 's' contains only one number and notlist is True.
    """
    out = [int(x) if x.isdigit() else float(x) for x in s.replace(', ', ',').split(',')]
    if any(isinstance(x, float) for x in out):
        out = list(map(float, out))
    if len(out) == 1 and notlist == True:
        out = out[0]
    return out

def csv_to_list(x: str, sumscores: bool = True) -> list:
    """
    Load a .csv file and turn it into a list or a list of lists.

    Parameters
    ----------
    x : str
        A string specifying path to .csv file.
    sumscores : bool
        Whether to create a single list with each entry being the sum of each
        sub-list.
    
    Returns
    -------
    list
        - If sumscores = False, a list containing lists of values.
        - If sumscores = True, a list containing the sum of each sub-list.
    """
    data = []
    with open(x, 'r') as file:
        reader = csv.reader(file)
        data = [row for row in list(map(float, reader))]
        #for row in reader:
        #    row = list(map(float, row))
        #    data.append(row)
    if sumscores:
        for i in range(len(data)):
            data[i] = sum(data[i])
    return data

def cronbachs_alpha(x: list) -> float:
    """
    Estimate Cronbach's Alpha reliability coefficient.

    Parameters
    ----------
    x : list
        A list of lists, where rows represent items and columns respondents.
    
    Returns
    -------
    float:
        Cronbach's Alpha reliability coefficient.
    """
    x = np.transpose(np.array(x))
    x = np.cov(x)
    n = x.shape[1]
    diag = sum(np.diag(x))
    var = sum(sum(x))
    alpha = (n / (n - 1)) * (1 - (diag / var))
    return alpha

def etl(mean: float, var: float, reliability: float, min: float = 0, max: float = 1) -> float:
    """
    Calculate the effective test length.

    Parameters
    ----------
    mean : float
        The mean of the score distribution.
    var : float
        The variance of the score distribution.
    reliability : float
        The reliability coefficient of the test-scores.
    min : float
        The minimum possible value of the scores.
    max : float
        The maximum possible value of the scores.
    
    Returns
    -------
    float:
        The effective test length.
    """
    return ((mean - min) * (max - mean) - (reliability * var)) / (var * (1 - reliability))

def k(mean: float, var: float, reliability: float, length: int) -> float:
    """
    Calculate Lord's k.

    Parameters
    ----------
    mean : float
        The mean of the score distribution.
    var : float
        The variance of the score distribution.
    reliability : float
        The reliability coefficient of the test-scores.
    length: int
        The length of the test (number of dichotomously scored items).

    Returns
    -------
    float:
        Lord's k.
    """
    vare = var * (1 - reliability)
    num = length * ((length - 1) * (var - vare) - length * var + mean * (length - mean))
    den = 2 * (mean * (length - mean) - (var - vare))
    return num / den

def dbeta4p(x: float, a: float, b: float, l: float, u: float) -> float:
    """
    Density function for the four-parameter beta distribution.

    Parameters
    ----------
    x : float
        Specified point along the four-parameter beta distribution.
    a : float
        The Alpha (first) shape parameter of the beta distribution.
    b : float
        The Beta (second) shape parameter of the beta distribution.
    l : float
        The lower-bound of the four-parameter beta distribution.
    u : float
        The upper-bound of the four-parameter beta distribution.

    Returns
    -------
    float:
        The probability density at a specific point along the four-parameter
        beta distribution.
    """
    if x < l or x > u:
        return 0
    else:
        return (1 / scipy.special.beta(a, b)) * (((x - l)**(a - 1) * (u - x)**(b - 1)) / (u - l)**(a + b - 1))

def rbeta4p(n: int, a: float, b: float, l: float = 0, u: float = 1) -> np.array[float]:
    """
    Random number generator for the four-parameter beta distribution.

    Parameters
    ----------
    n : int
        Number of random values to draw from the four-parameter beta distribution.    
    a : float
        The Alpha (first) shape parameter of the beta distribution.
    b : float
        The Beta (second) shape parameter of the beta distribution.
    l : float
        The lower-bound of the four-parameter beta distribution.
    u : float
        The upper-bound of the four-parameter beta distribution.

    Returns
    -------
    numpy array :
        An array of length n containing random values drawn from the four-parameter
        beta distribution.
    """
    return np.random.beta(a, b, n) * (u - l) + l

def beta4fit(x: list, moments: list = []) -> list[float]:
    """
    Fit a four-parameter beta distribution to a list of values.

    Parameters
    ----------
    x : list[float]
        List of values the distribution of which a four-parameter beta
        distribution is to be fit.
    moments : list[int]
        An optional list containing the first four moments of the distribution.
    
    Returns
    -------
    list[float]:
        A list containing the parameters of the beta distribution.
        - [0] : The Alpha (first) shape parameter.
        - [1] : The Beta (second) shape parameter.
        - [2] : The lower-bound.
        - [3] : The upper-bound.
    """
    if len(moments) == 0:
        m1 = stats.mean(x)
        s2 = stats.variance(x)
        x3 = list(x)
        x4 = list(x)
        for i in range(len(x)):
            x3[i] = ((x3[i] - m1)**3) / (s2**0.5)**3
            x4[i] = ((x4[i] - m1)**4) / (s2**0.5)**4
        g3 = (1 / len(x3)) * sum(x3)
        g4 = (1 / len(x4)) * sum(x4)
    else:
        m1 = moments[0]
        s2 = moments[1] - moments[0]**2
        g3 = (moments[2] - 3 * moments[0] * moments[1] + 2 * moments[0]**3) / ((s2**0.5)**3)
        g4 = (moments[3] - 4 * moments[0] * moments[2] + 6 * moments[0]**2 * moments[0] - 3 * moments[0]**3) / ((s2**0.5)**4)
    r = 6 * (g4 - g3**2 - 1) / (6 + 3 * g3**2 - 2 * g4)
    if g3 < 0:
        a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    else:
        b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
    l = m1 - ((a * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    u = m1 + ((b * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    return [a, b, l, u]

## Function for fitting a two-parameter beta distribution to a vector of values.
# x = vector of values.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
def beta2fit(x: list, l: float, u: float, moments: list = []) -> list[float]:
    """
    Fit a two-parameter beta distribution to a list of values.

    Parameters
    ----------
    x : list[float]
        List of values the distribution of which a four-parameter beta
        distribution is to be fit.
    moments : list[int]
        An optional list containing the first four moments of the distribution.
    
    Returns
    ------- 
    list[float]:
        A list containing the parameters of the beta distribution.
        - [0] : The Alpha (first) shape parameter.
        - [1] : The Beta (second) shape parameter.
        - [2] : The lower-bound.
        - [3] : The upper-bound.
    """
    if len(list) == 1:
        m1 = stats.mean(x)
        s2 = stats.variance(x)
    else:
        m1 = moments[0]
        s2 = moments[1] - moments[0]**2
    a = ((l - m1) * (l * (m1 - u) - m1**2 + m1 * u - s2)) / (s2 * (l - u))
    b = ((m1 - u) * (l * (u - m1) + m1**2 - m1 * u + s2)) / (s2 * (u - l))
    return [a, b, l, u]

def dcbinom(p: float, N: int, n: int, k: float) -> float:
    """
    Density function for Lord's two-term approximation of the compound binomial distribution.

    Parameters
    ----------
    p : float
        Probability of 'success'.
    N : int
        Total number of 'trials'.
    n : int
        Specific number of 'successes'.
    k : float
        Lord's k.

    Returns
    -------
    float:
        Probability of a specific number of 'successes' given N number of 'trials'.
    """
    a = binom.pmf(n, N, p)
    b = binom.pmf(n, N - 2, p)
    c = binom.pmf(n - 1, N - 2, p)
    d = binom.pmf(n - 2, N - 2, p)
    e = k * p * (1 - p)
    return a - e * (b - 2*c + d)

def dcbinom2(x: tuple, p: float, N: int, n: int, k: float, method: str) -> float:
    """
    Alternative density function for Lord's two-term approximation of the compound binomial distribution.

    Parameters
    ----------
    x : list or tuple
        tuple or list of choose-function outputs such as that produced from the choose_functions function.
    p : float
        Probability of 'success'.
    N : int
        Total number of 'trials'.
    n : int
        Specific number of 'successes'.
    k : float
        Lord's k (only necessary if method != 'll').
    method : str
        - "ll" for the Livingston and Lewis approach.
        - Any other string for the Hanson and Brennan approach.

    Returns
    -------
    float:
        Probability of a specific number of 'successes' given N number of 'trials'.
    """
    a = x[0]*(p**n)*(1 - p)**(N - n)
    if method != "ll":
        b = x[1]*(p**n)*(1 - p)**(N - n)
        c = x[2]*(p**n)*(1 - p)**(N - n)
        d = x[3]*(p**n)*(1 - p)**(N - n)
        e = k * p * (1 - p)
        return a - e * (b - 2*c + d)
    return a

def da_factorial(x: int):
    """
    Calculate factorial using direct arithmetic.

    Parameters
    ----------
    x : int
        Value to calculate the factorial of.

    Returns
    -------
    int:
        The factorial of 'x'.
    """
    if x > 0:
        return math.prod([i for i in range(1, x + 1)])
    elif x == 0:
        return 1
    else:
        return math.prod([i for i in range(x, 0)])

def choose_functions(N, n):
    """
    Choose functions for the compound beta-binomial distribution.

    Parameters
    ----------
    N : int
        Number of 'trials'.
    n : int
        Number of 'successes'.

    Returns
    -------
    tuple:
        A tuple of length 4 containing, in order, the output of the choose functions contained in the compound beta-binomial function.
    """
    def choose(N, n):
        return da_factorial(N) / (da_factorial(n) * da_factorial(N - n))
    a = choose(N, n)
    b = choose(N - 2, n)
    c = choose(N - 2, n - 1)
    d = choose(N - 2, n - 2)
    return (a, b, c, d)

def bbintegrate1(a: float, b: float, l: float, u: float, N: int, n: int, k: float, lower: float, upper: float, method: str = "ll", limit = 100) -> float:
    """
    Integrate across univariate beta-binomial distribution.

    Parameters
    ----------
    a : float
        The Alpha (first) shape parameter of the beta distribution.
    b : float
        The Beta (second) shape parameter of the beta distribution.
    l : float
        The lower-bound of the four-parameter beta distribution.
    u : float
        The upper-bound of the four-parameter beta distribution.
    N : int
        Number of 'trials'.
    n : int
        Number of 'successes'.
    k : float
        Lord's k (only necessary if method != 'll').
    lower : float
        The lower limit of the integral.
    upper : float
        The upper limit of the integral.
    method : str
        - "ll" for the Livingston and Lewis approach.
        - Any other string for the Hanson and Brennan approach.
    
    Returns
    -------
    float:
        Area under the curve at the specified interval for a beta-binomial distribution.
    """
    if method != "ll":
        def f(x, a, b, l, u, N, n, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n, k), limit = limit)
    else:
        def f(x, a, b, l, u, N, n):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n), limit = limit)
    
## Alternate integrate across univariate BB distribution.
# a = alpha shape parameter.
# b = beta shape parameter.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
# c = choose-function list or tuple (such as that produced by the choose_functions function).
# N = upper-bound of binomial distribution.
# n = specific binomial outcome.
# k = Lord's k.
# lower = lower-bound of integration.
# upper = upper bound of intergration.
# method = specify Livingston and Lewis ("LL") or Hanson and Brennan approach.
def bbintegrate1_2(a: float, b: float, l: float, u: float, c: tuple, N: int, n: int, k: float, lower: float, upper: float, method: str = "ll", limit = 100) -> float:
    if method != "ll":
        def f(x, a, b, l, u, c, N, n, k):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c, x, N, n, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c, N, n, k), limit = limit)
    else:
        def f(x, a, b, l, u, c, N, n):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c, x, N, n, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c, N, n), limit = limit)

## Integrate across bivariate BB distribution.
# a = alpha shape parameter.
# b = beta shape parameter.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
# N = upper-bound of binomial distribution.
# n1 = specific binomial outcome on first binomial trial.
# n2 = specific binomial outcome on second binomial trial.
# lower = lower-bound of integration.
# upper = upper bound of intergration.
# method = specify Livingston and Lewis ("LL") or Hanson and Brennan approach.
def bbintegrate2(a: float, b: float, l: float, u: float, N: int, n1: int, n2: int, k: float, lower: float, upper: float, method: str = "ll", limit = 100) -> float:
    if method != "ll":
        def f(x, a, b, l, u, N, n1, n2, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n1, k) * dcbinom(x, N, n2, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2, k), limit = limit)
    else:
        def f(x, a, b, l, u, N, n1, n2):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n1, N, x) * binom.pmf(n2, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2), limit = limit)

## Integrate across bivariate BB distribution.
# a = alpha shape parameter.
# b = beta shape parameter.
# l = lower-bound location parameter.
# u = upper-bound location parameter.
# c = choose-function list or tuple (such as that produced by the choose_functions function).
# N = upper-bound of binomial distribution.
# n1 = specific binomial outcome on first binomial trial.
# n2 = specific binomial outcome on second binomial trial.
# lower = lower-bound of integration.
# upper = upper bound of intergration.
# method = specify Livingston and Lewis ("ll") or Hanson and Brennan approach (anything but "ll").
def bbintegrate2_1(a: float, b: float, l: float, u: float, c1: tuple, c2: tuple, N: int, n1: int, n2: int, k: float, lower: float, upper: float, method: str = "ll", limit = 100) -> float:
    if method != "ll":
        def f(x, a, b, l, u, c1, c2, N, n1, n2, k):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c1, x, N, n1, k, method) * dcbinom2(c2, x, N, n2, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2, k), limit = limit)
    else:
        def f(x, a, b, l, u, c1, c2, N, n1, n2):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c1, x, N, n1, k, method) * dcbinom2(c2, x, N, n2, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2), limit = limit)


## Function for calculating the descending factorial each value of a vector.
# x = vector of values.
# r = the power x is to be raised to.
def dfac(x: list, r = int):
    x1 = list(x)
    for i in range(len(x)):
        if r <= 1:
            x1[i] = x1[i]**r
        else:
            for j in range(1, r + 1):
                if j > 1:
                    x1[i] = x1[i] * (x[i] - j + 1)
    return x1

## Function for calculating the first four raw moments of the true-score distribution.
# x = vector of values.
# n = the effective or actual test length.
# k = Lord's k.
def tsm(x: list, n: int, k: float):
    m = [0, 0, 0, 0]
    for i in range(0, 4):
        if i == 0:
            m[i] = stats.mean(x) / n
        else:
            M = i + 1
            a = (dfac([n], 2)[0] + k * dfac([M], 2)[0])
            b = stats.mean(dfac(x, M)) / dfac([n - 2], M - 2)[0]
            c = k * dfac([M], 2)[0] * m[i]
            m[i] = (b / a) + c
    return m

## Estimate the true-score 2 or 4 parameter beta distribution parameters.
# x = vector of values
# n = actual or effective test length.
# k = Lord's k.
# model = whether 2 or 4 parameters are to be fit.
# l = if model = 2, specified lower-bound of 2-parameter distribution.
# u = if model = 2, specified upper-bound of 2-parameter distribution.
def betaparameters(x: list, n: int, k: float, model: int = 4, l: float = 0, u: float = 1):
    m = tsm(x, n, k)
    s2 = m[1] - m[0]**2
    g3 = (m[2] - 3 * m[0] * m[1] + 2 * m[0]**3) / (math.sqrt(s2)**3)
    g4 = (m[3] - 4 * m[0] * m[2] + 6 * m[0]**2 * m[1] - 3 * m[0]**4) / (math.sqrt(s2)**4)
    if model == 4:
        r = 6 * (g4 - g3**2 - 1) / (6 + 3 * g3**2 - 2 * g4)
        if g3 < 0:
            a = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
            b = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        else:
            b = r / 2 * (1 + (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
            a = r / 2 * (1 - (1 - ((24 * (r + 1)) / ((r + 2) * (r + 3) * g4 - 3 * (r - 6) * (r + 1))))**0.5)
        l = m[0] - ((a * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
        u = m[0] + ((b * (s2 * (a + b + 1))**0.5) / (a * b)**0.5)
    if model == 2:
        a = ((l - m[0]) * (l * (m[0] - u) - m[0]**2 + m[0] * u - s2)) / (s2 * (l - u))
        b = ((m[0] - u) * (l * (u - m[0]) + m[0]**2 - m[0] * u + s2)) / (s2 * (u - l))
    return {"alpha":  a, "beta": b, "l": l, "u": u}