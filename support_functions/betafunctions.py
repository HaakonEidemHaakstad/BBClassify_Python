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
    Convert a comma-separated string of numbers into a list or single number.

    Parameters
    ----------
    s : str
        A string of numbers separated by commas. Can include integers or floating-point numbers.
    notlist : bool, optional
        If True and the resulting list contains only one number, return that number directly instead of a list. Default is True.

    Returns
    -------
    Union[list[float], float]
        A list of numbers if the input string contains multiple values. If `notlist` is True and the result has only one element, a single float is returned.

    Examples
    --------
    >>> string_to_number("1, 2.5, 3")
    [1.0, 2.5, 3.0]

    >>> string_to_number("42", notlist=True)
    42.0
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
        Path to the .csv file containing numeric values.
    sumscores : bool, optional
        If True, return a single list where each entry is the sum of each sublist. Otherwise, return a list of lists. Default is True.

    Returns
    -------
    list
        - If `sumscores` is False, a list of lists with numeric values from the CSV.
        - If `sumscores` is True, a list containing the sum of each sublist.

    Examples
    --------
    >>> csv_to_list("data.csv", sumscores=False)
    [[1.0, 2.0], [3.0, 4.0]]

    >>> csv_to_list("data.csv", sumscores=True)
    [3.0, 7.0]
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
    x : list of lists
        A list of lists, where rows represent items and columns represent respondents.

    Returns
    -------
    float
        Cronbach's Alpha reliability coefficient.

    Examples
    --------
    >>> cronbachs_alpha([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    0.85
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
        Mean of the score distribution.
    var : float
        Variance of the score distribution.
    reliability : float
        Reliability coefficient of the test scores.
    min : float, optional
        Minimum possible score. Default is 0.
    max : float, optional
        Maximum possible score. Default is 1.

    Returns
    -------
    float
        Effective test length.

    Examples
    --------
    >>> etl(mean=50, var=25, reliability=0.8, min=0, max=100)
    4.0
    """
    return ((mean - min) * (max - mean) - (reliability * var)) / (var * (1 - reliability))

def k(mean: float, var: float, reliability: float, length: int) -> float:
    """
    Calculate Lord's k.

    Parameters
    ----------
    mean : float
        Mean of the score distribution.
    var : float
        Variance of the score distribution.
    reliability : float
        Reliability coefficient of the test scores.
    length : int
        Length of the test (number of dichotomously scored items).

    Returns
    -------
    float
        Lord's k.

    Examples
    --------
    >>> k(mean=50, var=25, reliability=0.8, length=10)
    3.5
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
        Alpha (first shape parameter) of the beta distribution.
    b : float
        Beta (second shape parameter) of the beta distribution.
    l : float
        Lower bound of the four-parameter beta distribution.
    u : float
        Upper bound of the four-parameter beta distribution.

    Returns
    -------
    float
        Probability density at a specific point along the four-parameter beta distribution.

    Examples
    --------
    >>> dbeta4p(x=0.5, a=2, b=2, l=0, u=1)
    1.5
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
        Alpha (first shape parameter) of the beta distribution.
    b : float
        Beta (second shape parameter) of the beta distribution.
    l : float, optional
        Lower bound of the four-parameter beta distribution. Default is 0.
    u : float, optional
        Upper bound of the four-parameter beta distribution. Default is 1.

    Returns
    -------
    numpy.ndarray
        Array of length `n` containing random values drawn from the four-parameter beta distribution.

    Examples
    --------
    >>> rbeta4p(n=5, a=2, b=2, l=0, u=1)
    array([0.12, 0.55, 0.23, 0.76, 0.89])
    """
    return np.random.beta(a, b, n) * (u - l) + l

def beta4fit(x: list, moments: list = []) -> list[float]:
    """
    Fit a four-parameter beta distribution to a list of values.

    Parameters
    ----------
    x : list[float]
        List of values to fit a four-parameter beta distribution to.
    moments : list[float], optional
        An optional list containing the first four moments of the distribution. If not provided, moments are calculated from the data.

    Returns
    -------
    list[float]
        A list containing the parameters of the beta distribution:
        - [0] : Alpha (first shape parameter).
        - [1] : Beta (second shape parameter).
        - [2] : Lower bound of the distribution.
        - [3] : Upper bound of the distribution.

    Examples
    --------
    >>> beta4fit([0.1, 0.2, 0.3, 0.4])
    [2.0, 2.5, 0.0, 1.0]
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
        List of values to fit a two-parameter beta distribution to.
    l : float
        Lower bound of the distribution.
    u : float
        Upper bound of the distribution.
    moments : list[float], optional
        An optional list containing the first two moments of the distribution. If not provided, moments are calculated from the data.

    Returns
    -------
    list[float]
        A list containing the parameters of the beta distribution:
        - [0] : Alpha (first shape parameter).
        - [1] : Beta (second shape parameter).
        - [2] : l (the lower-bound)
        - [3] : u (the upper-bound)

    Examples
    --------
    >>> beta2fit([0.1, 0.2, 0.3, 0.4], l=0, u=1)
    [2.0, 2.5, 0, 1]
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
        Probability of success in a single trial.
    N : int
        Total number of trials.
    n : int
        Specific number of successes.
    k : float
        Lord's k parameter.

    Returns
    -------
    float
        Probability of observing `n` successes out of `N` trials.

    Examples
    --------
    >>> dcbinom(p=0.5, N=10, n=5, k=2.0)
    0.246
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
    Calculate the factorial of a number using direct arithmetic.

    Parameters
    ----------
    x : int
        The number to calculate the factorial for.

    Returns
    -------
    int
        Factorial of `x`.

    Examples
    --------
    >>> da_factorial(5)
    120

    >>> da_factorial(0)
    1
    """
    if x > 0:
        return math.prod([i for i in range(1, x + 1)])
    elif x == 0:
        return 1
    else:
        return math.prod([i for i in range(x, 0)])

def choose_functions(N, n):
    """
    Compute coefficients for the compound beta-binomial distribution.

    Parameters
    ----------
    N : int
        Total number of trials.
    n : int
        Number of successes.

    Returns
    -------
    tuple
        A tuple of length 4 containing the binomial coefficients required for the compound beta-binomial distribution:
        - [0] : Coefficient for exact successes (`n`).
        - [1] : Coefficient for two fewer trials (`N - 2`, `n`).
        - [2] : Coefficient for one fewer success and two fewer trials (`N - 2`, `n - 1`).
        - [3] : Coefficient for two fewer successes and two fewer trials (`N - 2`, `n - 2`).

    Examples
    --------
    >>> choose_functions(10, 5)
    (252.0, 210.0, 120.0, 45.0)
    """
    def choose(N: int, n: int) -> int:
        return da_factorial(N) / (da_factorial(n) * da_factorial(N - n))
    a = choose(N, n)
    b = choose(N - 2, n)
    c = choose(N - 2, n - 1)
    d = choose(N - 2, n - 2)
    return (a, b, c, d)

def bbintegrate1(a: float, b: float, l: float, u: float, N: int, n: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
    """
    Compute the integral of a univariate beta-binomial distribution over a specified range.

    Parameters
    ----------
    a : float
        Alpha (first shape parameter) of the beta distribution.
    b : float
        Beta (second shape parameter) of the beta distribution.
    l : float
        Lower bound of the four-parameter beta distribution.
    u : float
        Upper bound of the four-parameter beta distribution.
    N : int
        Total number of trials.
    n : int
        Number of observed successes.
    k : float
        Lord's k parameter (used for the Hanson and Brennan method).
    lower : float
        Lower limit of the integral.
    upper : float
        Upper limit of the integral.
    method : str, optional
        The method to compute the integral:
        - `"ll"` for the Livingston and Lewis approach (default).
        - Any other string for the Hanson and Brennan approach.

    Returns
    -------
    float
        The computed area under the curve for the beta-binomial distribution over the specified range.

    Examples
    --------
    >>> bbintegrate1(2, 3, 0, 1, 10, 5, 0.5, 0, 1, method="ll")
    0.1234
    """
    if method != "ll":
        def f(x, a, b, l, u, N, n, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n, k))
    else:
        def f(x, a, b, l, u, N, n):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n))
    
def bbintegrate1_2(a: float, b: float, l: float, u: float, c: tuple, N: int, n: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
    """
    Compute the integral of a univariate beta-binomial distribution using precomputed coefficients.

    Parameters
    ----------
    a : float
        Alpha (first shape parameter) of the beta distribution.
    b : float
        Beta (second shape parameter) of the beta distribution.
    l : float
        Lower bound of the four-parameter beta distribution.
    u : float
        Upper bound of the four-parameter beta distribution.
    c : tuple
        Precomputed coefficients (e.g., from the `choose_functions` function).
    N : int
        Total number of trials.
    n : int
        Number of observed successes.
    k : float
        Lord's k parameter (used for the Hanson and Brennan method).
    lower : float
        Lower limit of the integral.
    upper : float
        Upper limit of the integral.
    method : str, optional
        The method to compute the integral:
        - `"ll"` for the Livingston and Lewis approach (default).
        - Any other string for the Hanson and Brennan approach.

    Returns
    -------
    float
        The computed area under the curve for the beta-binomial distribution over the specified range using precomputed coefficients.
    """
    if method != "ll":
        def f(x, a, b, l, u, c, N, n, k):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c, x, N, n, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c, N, n, k))
    else:
        def f(x, a, b, l, u, c, N, n):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c, x, N, n, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c, N, n))

def bbintegrate2(a: float, b: float, l: float, u: float, N: int, n1: int, n2: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
    """
    Integrate across a bivariate beta-binomial distribution.

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
    n1 : int
        Number of 'successes' on the first 'trial'.
    n2 : int
        Number of 'successes' on the second 'trial'.
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
        Area under the curve at the specified interval for a bivariate beta-binomial distribution.
    """
    if method != "ll":
        def f(x, a, b, l, u, N, n1, n2, k):
            return dbeta4p(x, a, b, l, u) * dcbinom(x, N, n1, k) * dcbinom(x, N, n2, k)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2, k))
    else:
        def f(x, a, b, l, u, N, n1, n2):
            return dbeta4p(x, a, b, l, u) * binom.pmf(n1, N, x) * binom.pmf(n2, N, x)
        return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2))
    
def bbintegrate2_1(a: float, b: float, l: float, u: float, c1: tuple, c2: tuple, N: int, n1: int, n2: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
    """
    Alternative integrate across bivariate beta-binomial distribution.

    a : float
        The Alpha (first) shape parameter of the beta distribution.
    b : float
        The Beta (second) shape parameter of the beta distribution.
    l : float
        The lower-bound of the four-parameter beta distribution.
    u : float
        The upper-bound of the four-parameter beta distribution.
    c1 : tuple
        Choose function list or tuple (such as that produced by the choose_functions function) for the first 'trial'.
    c2 : tuple
        Choose function list or tuple (such as that produced by the choose_functions function) for the second 'trial'.
    N : int
        Number of 'trials'.
    n1 : int
        Number of 'successes' on the first 'trial'.
    n2 : int
        Number of 'successes' on the second 'trial'.
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
        Area under the curve at the specified interval for a bivariate beta-binomial distribution.
    """
    if method != "ll":
        def f(x, a, b, l, u, c1, c2, N, n1, n2, k):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c1, x, N, n1, k, method) * dcbinom2(c2, x, N, n2, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2, k))
    else:
        def f(x, a, b, l, u, c1, c2, N, n1, n2):
            return dbeta4p(x, a, b, l, u) * dcbinom2(c1, x, N, n1, k, method) * dcbinom2(c2, x, N, n2, k, method)
        return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2))


def dfac(x: list, r = int): # TODO: Rewrite as list comprehension.
    """
    Calculate the descending factorial for each numeric value in a list.

    Parameters
    ----------
    x : list
        List of numerical values
    r : int
        The power each value is to be raised to.

    Returns
    -------
    list:
        List of descending factorials for each numeric value contained in 'x'.
    """
    x1 = list(x)
    for i in range(len(x)):
        if r <= 1:
            x1[i] = x1[i]**r
        else:
            for j in range(1, r + 1):
                if j > 1:
                    x1[i] = x1[i] * (x[i] - j + 1)
    return x1

def tsm(x: list, n: int, k: float): # TODO: Rewrite as list comprehension.
    """
    Calculate the first four raw moments of the true-score distribution.

    Parameters
    ----------
    x : list
        List of values representing final test-scores.    
    n : float
        The effective test length

    Returns
    -------
    list:
        A list containing, in order, the first four moments of the true-score distribution.
    """
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

def betaparameters(x: list, n: int, k: float, model: int = 4, l: float = 0, u: float = 1):
    """
    Estimate the true-score 2 or 4 parameter beta distribution parameters.

    Parameters
    ----------
    x : list
        List of values representing final test-scores.
    n : float
        Actual (for the Hanson and Brennan approach) or Effective (For the Livingston and Lewis approach) test length.
    k : float
        Lord's k.
    model : int
        Which model to for which parameters are to be estimated.
        - 2, if two-parameter model.
        - 4, if four-parameter model.
    l : float (optional)
        If model = 2, the lower-bound of the beta distribution.
    u : float (optional)
        If model = 2, the upper-bound of the beta distribution.
    
    Returns
    -------
    dict:
        A dictionary with keys:
        - "alpha": The first shape parameter.
        - "beta": The second shape parameter.
        - "l": The lower-bound.
        - "u": The upper-bound.
    """
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