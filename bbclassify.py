from warnings import warn
import scipy.special
from scipy.integrate import quad
from scipy.stats import binom, chi2
import statistics as stats
import math
import numpy as np

class bbclassify():
    def __init__(self, data: list, reliability: float, min_score: float, max_score: float, cut_scores: list[float], method: str = "ll", model: int = 4, l: float = 0, u: float = 1, failsafe: bool = False):
        """
        Estimate the parameters of the beta-binomial models.

        This class supports two approaches:
        - Livingston and Lewis (LL)
        - Hanson and Brennan (HB)

        Parameters
        ----------
        x : list[float] or dict
            Either a list of values to which a beta-binomial model is to be fitted,
            or a dictionary of model parameters.
        reliability : float
            The test-score reliability coefficient.
        min_score : float
            Minimum possible test score (used only in the Livingston and Lewis approach).
        max_score : float
            - For the Livingston and Lewis approach: Maximum possible test score.
            - For the Hanson and Brennan approach: Actual test length (number of items).
        cut_scores : list[float]
            List of cut-scores.
        method : str
            Estimation approach to use:
            - "ll": Livingston and Lewis (default)
            - Any other string: Hanson and Brennan
        model : int, optional
            The type of beta-binomial model to fit. Use "2" for a two-parameter model
            or "4" for a four-parameter model. Default is 4.
        l : float, optional
            Lower bound for the beta true-score distribution in a two-parameter model.
            Must be between 0 and 1, and less than `u`. Default is 0.
        u : float, optional
            Upper bound for the beta true-score distribution in a two-parameter model.
            Must be between 0 and 1, and greater than `l`. Default is 1.
        failsafe : bool, optional
            If True, reverts to a two-parameter model if the four-parameter model fails
            or produces invalid estimates. Default is False.
        """
        self.data = data
        self.reliability = reliability
        self.min_score = min_score
        self.max_score = max_score
        self.cut_scores = [self.min_score] + cut_scores + [self.max_score]
        self.cut_truescores = [(i - self.min_score) / (self.max_score + self.min_score) for i in self.cut_scores]
        self.method = method
        self.model = model
        self.l = l
        self.u = u
        self.failsafe = failsafe


        self.Modelfit_chi_squared = "Model fit not yet estimated. Run .modelfit() to estimate model fit."
        self.Modelfit_degrees_of_freedom = "Model fit not yet estimated. Run .modelfit() to estimate model fit."
        self.Modelfit_p_value = "Model fit not yet estimated. Run .modelfit() to estimate model fit."
        self.Accuracy = "Accuracy not yet estimated. Run .accuracy() to estimate model fit."
        self.Consistency = "Consistency not yet estimated, Run .consistency() to estiamte model fit."

        if isinstance(self.data, dict): # Parameters do not have to be estimated if a dict of parameter values is supplied.
            self.parameters = self.data
            if self.method == "ll":
                self.N = self.parameters["etl"]
            else:
                self.N = self.parameters["atl"]
        else: # If raw data is supplied, estimate parameters from the data.
            if self.method == "ll": # For the Livingston and Lewis method:
                self.effective_test_length = self._calculate_etl(stats.mean(self.data), stats.variance(self.data), self.reliability, self.min_score, self.max_score)
                self.N = round(self.effective_test_length)
                self.K = 0
                self.parameters = self._betaparameters(self.data, self.N, 0, self.model, self.l, self.u)
                self.parameters["etl"] = self.effective_test_length
                self.parameters["etl rounded"] = self.N
                self.parameters["lords_k"] = 0
            else: # For the Hanson and Brennan method:
                self.N = self.max_score
                self.K = self._calculate_lords_k(stats.mean(self.data), stats.variance(self.data), self.reliability, self.N)
                self.parameters = self._betaparameters(self.data, self.N, self.K, self.model, self.l, self.u)
                self.parameters["lords_k"] = self.K
            # If a four-parameter fitting procedure produced invalid location parameter estimates, and the failsafe was specified to
            # engage in such a circumstance, fit a two-parameter model instead with the location-parameters specified by the user.
            if (self.failsafe == True and self.model == 4) and (self.parameters["l"] < 0 or self.parameters["u"] > 1):
                warn(f"Failsafe triggered. True-score fitting procedure produced impermissible parameter values (l = {self.parameters['l']}, u = {self.parameters['u']}).")
                self.parameters = self._betaparameters(self.data, self.N, self.K, 2, self.l, self.u)
                self.model = 2

        self.choose_values = [self._choose_functions(self.N, i) for i in range(self.N + 1)]
    
    # Function for testing model fit.
    def modelfit(self):
        # If the input to the data argument is a dictionary of model-parameters, throw a value error.
        if isinstance(self.data, dict):
            raise ValueError("Model fit testing requires observed test-scores as data input.")
        n_respondents = len(self.data)
        # Make a list of the model-implied (expected) frequency-distribution.
        expected = [
                    self._bbintegrate1_1(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"],
                            self.choose_values[i], self.N, i, self.parameters["lords_k"], 0, 1, self.method)[0] * n_respondents 
                            for i in range(self.N + 1)
                    ]
        # Make a list of the actual observed-score frequency-distribution.
        # If method == "ll", rescale observed-scores onto the effective test-length scale, and round to integer for count.
        if self.method == "ll": 
            observed = [round(((i - self.min_score) / (self.max_score - self.min_score)) * self.N) for i in self.data]
            observed = [observed.count(i) for i in range(self.N + 1)]
        else: # Otherwise, simply count frequencies of each score from 0 to N items.
            observed = [self.data.count(i) for i in range(self.N + 1)]

        # Collapse (first left-to-right, then right-to-left) cells with expected values less than 1.
        for _ in range(2):
            length = len(expected)
            for i in range(length):
                # If the next cell has an expected value less than 1.
                if expected[i + 1] < 1:
                    # Add the value of the current cell to the next cell,
                    expected[i + 1] += expected[i]
                    observed[i + 1] += observed[i]
                    # And set the value of the current cell to None for deletion.
                    expected[i], observed[i] = None, None
                else:
                    # Terminate loop if the next cell has an expected value greater than or equal to 1.
                    break

            # Retain observed-value cells that are not equal to None ...
            observed = [j for j in observed if j is not None][::-1] # Reverse the list for next iteration.
            # ... and retain expected-value cells that are not equal to None.
            expected = [j for j in expected if j is not None][::-1] # Reverse the list for next iteration.
        
        # Rounding errors will make the sum of the expected frequencies not add up exactly to observed frequencies.
        # Therefore, normalize the frequencies by multiplying all expected frequencies by a normalizing constant.
        # This ensures that the sum of expected frequencies are the same as the sum of observed frequencies.
        normalize = sum(observed) / sum(expected)
        expected = [j*normalize for j in expected]

        # Calculate chi-squared, degrees of freedom, and p-value.
        self.Modelfit_chi_squared = sum([(observed[i] - expected[i])**2 / expected[i] for i in range(len(expected))])
        self.Modelfit_degrees_of_freedom = len(expected) - self.model
        self.Modelfit_p_value = 1 - chi2.cdf(self.modelfit_chi_squared, self.modelfit_degrees_of_freedom)
    
    # Function for estimating classification accuracy.
    def accuracy(self):
        confmat = np.zeros((self.N + 1, len(self.cut_scores) - 1))
        for i in range(len(self.cut_scores) - 1):
            for j in range(self.N + 1):
                confmat[j, i] = self._bbintegrate1_1(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"], 
                                               self.choose_values[j], self.N, j, self.parameters["lords_k"], self.cut_truescores[i], self.cut_truescores[i + 1], self.method)[0]
        self.confusionmatrix = np.zeros((len(self.cut_scores) - 1, len(self.cut_scores) - 1))
        for i in range(len(self.cut_scores) - 1):
            for j in range(len(self.cut_scores) - 1):
                if i != len(self.cut_scores) - 2:
                    self.confusionmatrix[i, j] = sum(confmat[self.cut_scores[i]:self.cut_scores[i + 1], j]) 
                else:
                    self.confusionmatrix[i, j] = sum(confmat[self.cut_scores[i]:, j])
        self.Accuracy = sum([self.confusionmatrix[i, i] for i in range(len(self.cut_scores) - 1)])
        return self

    # Function for estimating classification consistency.
    def consistency(self):
        consmat = np.zeros((self.N + 1, self.N + 1))
        for i in range(self.N + 1):
            for j in range(self.N + 1):
                if i <= j:
                    consmat[i, j] = self._bbintegrate2_1(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"], 
                                                   self.choose_values[i], self.choose_values[j], self.N, i, j, self.parameters["lords_k"], 0, 1, self.method)[0]
        lower_triangle = np.tril_indices(consmat.shape[0], 0)
        consmat[lower_triangle] = consmat.T[lower_triangle]
        self.consistencymatrix = np.zeros((len(self.cut_scores) - 1, len(self.cut_scores) - 1))
        for i in range(len(self.cut_scores) - 1):
            for j in range(len(self.cut_scores) - 1):
                if i == 0 and j == 0:
                    self.consistencymatrix[i, j] = sum(sum(consmat[0:self.cut_scores[i + 1], 0:self.cut_scores[j + 1]]))
                elif i == 0 and (j != 0 and j != len(self.cut_scores) - 2):
                    self.consistencymatrix[i, j] = sum(sum(consmat[0:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1]]))
                elif i == 0  and j == len(self.cut_scores) - 2:
                    self.consistencymatrix[i, j] = sum(sum(consmat[0:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1] + 1]))
                elif (i != 0 and i != len(self.cut_scores) - 2) and j == 0:
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1], 0:self.cut_scores[j + 1]]))
                elif (i != 0 and i != len(self.cut_scores) - 2) and (j != 0 and j != len(self.cut_scores) - 2):
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1]]))
                elif (i != 0 and i != len(self.cut_scores) - 2) and j == len(self.cut_scores) - 2:
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1] + 1]))
                elif i == len(self.cut_scores) - 2 and j == 0:
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1] + 1, 0:self.cut_scores[j + 1]]))
                elif i == len(self.cut_scores) - 2 and (j != 0 and j != len(self.cut_scores) - 2):
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1] + 1, self.cut_scores[j]:self.cut_scores[j + 1]]))
                else:
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1] + 1, self.cut_scores[j]:self.cut_scores[j + 1] + 1]))
        self.Consistency = sum([self.consistencymatrix[i, i] for i in range(len(self.cut_scores) - 1)])
        return self
        
    def _calculate_etl(self, mean: float, var: float, reliability: float, min: float = 0, max: float = 1) -> float:
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
        >>> _calculate_etl(mean = 50, var = 25, reliability = 0.8, min = 0, max = 100)
        4.0
        """
        return ((mean - min) * (max - mean) - (reliability * var)) / (var * (1 - reliability))
    def _calculate_lords_k(self, mean: float, var: float, reliability: float, length: int) -> float:
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
        >>> _calculate_lords_k(mean = 50, var = 25, reliability = 0.8, length = 10)
        3.5
        """
        vare = var * (1 - reliability)
        num = length * ((length - 1) * (var - vare) - length * var + mean * (length - mean))
        den = 2 * (mean * (length - mean) - (var - vare))
        return num / den

    def _dbeta4p(self, x: float, a: float, b: float, l: float, u: float) -> float:
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
        >>> _dbeta4p(x = 0.5, a = 2, b = 2, l = 0, u = 1)
        1.5
        """
        if x < l or x > u:
            return 0
        else:
            return (1 / scipy.special.beta(a, b)) * (((x - l)**(a - 1) * (u - x)**(b - 1)) / (u - l)**(a + b - 1))

    def _beta4fit(self, x: list, moments: list = []) -> list[float]:
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
        >>> _beta4fit([0.1, 0.2, 0.3, 0.4])
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
    def _beta2fit(self, x: list, l: float, u: float, moments: list = []) -> list[float]:
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
        >>> _beta2fit([0.1, 0.2, 0.3, 0.4], l = 0, u = 1)
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

    def _dcbinom(self, p: float, N: int, n: int, k: float) -> float:
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
            Probability of observing 'n' successes out of 'N' trials.

        Examples
        --------
        >>> _dcbinom(p = 0.5, N = 10, n = 5, k = 2.0)
        0.246
        """
        a = binom.pmf(n, N, p)
        b = binom.pmf(n, N - 2, p)
        c = binom.pmf(n - 1, N - 2, p)
        d = binom.pmf(n - 2, N - 2, p)
        e = k * p * (1 - p)
        return a - e * (b - 2*c + d)

    def _dcbinom2(self, x: tuple, p: float, N: int, n: int, k: float, method: str) -> float:
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

    def _da_factorial(self, x: int):
        """
        Calculate the factorial of a number using direct arithmetic.

        Parameters
        ----------
        x : int
            The number to calculate the factorial for.

        Returns
        -------
        int
            Factorial of 'x'.

        Examples
        --------
        >>> _da_factorial(5)
        120

        >>> _da_factorial(0)
        1
        """
        if x > 0:
            return math.prod([i for i in range(1, x + 1)])
        elif x == 0:
            return 1
        else:
            return math.prod([i for i in range(x, 0)])
    
    def _choose(self, N: int, n: int) -> int:
        """
        Compute the binomial coefficients for 'n' successes over 'N' trials.

        Parameters
        ----------
        N : int
            Total number of trials.
        n : int
            Number of successes.
        
        Returns
        -------
        int
            Coefficient for exact successes ('n') for 'N' trials.
        
        Examples
        --------
        >>> _choose(10, 5)
        252
        """
        return self._da_factorial(N) / (self._da_factorial(n) * self._da_factorial(N - n))

    def _choose_functions(self, N, n):
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
            - [0] : Coefficient for exact successes ('n').
            - [1] : Coefficient for two fewer trials ('N - 2', 'n').
            - [2] : Coefficient for one fewer success and two fewer trials ('N - 2', 'n - 1').
            - [3] : Coefficient for two fewer successes and two fewer trials ('N - 2', 'n - 2').

        Examples
        --------
        >>> _choose_functions(10, 5)
        (252.0, 210.0, 120.0, 45.0)
        """        
        a = self._choose(N, n)
        b = self._choose(N - 2, n)
        c = self._choose(N - 2, n - 1)
        d = self._choose(N - 2, n - 2)
        return (a, b, c, d)

    def _bbintegrate1(self, a: float, b: float, l: float, u: float, N: int, n: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
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
            - 'll' for the Livingston and Lewis approach (default).
            - Any other string for the Hanson and Brennan approach.

        Returns
        -------
        float
            The computed area under the curve for the beta-binomial distribution over the specified range.

        Examples
        --------
        >>> _bbintegrate1(2, 3, 0, 1, 10, 5, 0.5, 0, 1, method = 'll')
        0.1234
        """
        if method != "ll":
            def f(x, a, b, l, u, N, n, k):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom(x, N, n, k)
            return quad(f, lower, upper, args = (a, b, l, u, N, n, k))
        else:
            def f(x, a, b, l, u, N, n):
                return self._dbeta4p(x, a, b, l, u) * binom.pmf(n, N, x)
            return quad(f, lower, upper, args = (a, b, l, u, N, n))
        
    def _bbintegrate1_1(self, a: float, b: float, l: float, u: float, c: tuple, N: int, n: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
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
            Precomputed coefficients (e.g., from the '_choose_functions' function).
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
            - 'll' for the Livingston and Lewis approach (default).
            - Any other string for the Hanson and Brennan approach.

        Returns
        -------
        float
            The computed area under the curve for the beta-binomial distribution over the specified range using precomputed coefficients.
        """
        if method != "ll":
            def f(x, a, b, l, u, c, N, n, k):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom2(c, x, N, n, k, method)
            return quad(f, lower, upper, args = (a, b, l, u, c, N, n, k))
        else:
            def f(x, a, b, l, u, c, N, n):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom2(c, x, N, n, k, method)
            return quad(f, lower, upper, args = (a, b, l, u, c, N, n))

    def _bbintegrate2(self, a: float, b: float, l: float, u: float, N: int, n1: int, n2: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
        """
        Compute the integral of a bivariate beta-binomial distribution over a specified range.

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
            - 'll' for the Livingston and Lewis approach.
            - Any other string for the Hanson and Brennan approach.

        Returns
        -------
        float:
            The computed area under the curve for the bivariate beta-binomial distribution over the specified range using precomputed coefficients.
        """
        if method != "ll":
            def f(x, a, b, l, u, N, n1, n2, k):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom(x, N, n1, k) * self._dcbinom(x, N, n2, k)
            return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2, k))
        else:
            def f(x, a, b, l, u, N, n1, n2):
                return self._dbeta4p(x, a, b, l, u) * binom.pmf(n1, N, x) * binom.pmf(n2, N, x)
            return quad(f, lower, upper, args = (a, b, l, u, N, n1, n2))
        
    def _bbintegrate2_1(self, a: float, b: float, l: float, u: float, c1: tuple, c2: tuple, N: int, n1: int, n2: int, k: float, lower: float, upper: float, method: str = "ll") -> float:
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
        c1 : tuple
            Precomputed coefficients (e.g., from the '_choose_functions' function).
        c2 : tuple
            Precomputed coefficients (e.g., from the '_choose_functions' function).
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
            - 'll' for the Livingston and Lewis approach (default).
            - Any other string for the Hanson and Brennan approach.

        Returns
        -------
        float
            The computed area under the curve for the beta-binomial distribution over the specified range using precomputed coefficients.
        """
        if method != "ll":
            def f(x, a, b, l, u, c1, c2, N, n1, n2, k):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom2(c1, x, N, n1, k, method) * self._dcbinom2(c2, x, N, n2, k, method)
            return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2, k))
        else:
            def f(x, a, b, l, u, c1, c2, N, n1, n2):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom2(c1, x, N, n1, k, method) * self._dcbinom2(c2, x, N, n2, k, method)
            return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2))


    def _dfac(self, x: list, r = int): # TODO: Rewrite as list comprehension.
        """
        Calculate the descending factorial for each numeric value in a list.

        The descending factorial of a number 'n' to power 'r' is calculated as:
        'n * (n - 1) * (n - 2) * ... * (n - (r - 1))'.

        Parameters
        ----------
        x : list
            List of numeric values for which the descending factorial will be calculated.
        r : int
            Number of terms in the descending factorial. If `r = 1`, the function simply raises each value in `x` to the power of `r`.

        Returns
        -------
        list
            A list containing the descending factorials for each value in the input list.

        Examples
        --------
        >>> _dfac([5, 6, 7], 3)
        [60, 120, 210]

        >>> _dfac([4, 3], 1)
        [4, 3]
        """
        x1 = x.copy()
        for i in range(len(x)):
            if r <= 1:
                x1[i] = x1[i]**r
            else:
                for j in range(1, r + 1):
                    if j > 1:
                        x1[i] = x1[i] * (x[i] - j + 1)
        return x1

    def _tsm(self, x: list, n: int, k: float): # TODO: Refactor as list comprehension.
        """
        Calculate the first four raw moments of the true-score distribution.

        Parameters
        ----------
        x : list
            List of values representing final test scores.
        n : int
            Effective test length.
        k : float
            Lord's k parameter, used for adjusting the moments based on test characteristics.

        Returns
        -------
        list
            A list containing the first four raw moments of the true-score distribution, in order:
            - [0] : The mean.
            - [1] : The second raw moment.
            - [2] : The third raw moment.
            - [3] : The fourth raw moment.

        Examples
        --------
        >>> _tsm([5, 6, 7, 8], 10, 0.2)
        [0.65, 0.422, 0.276, 0.182]
        """
        m = [stats.mean(x) / n, 0, 0, 0]
        for i in range(1, 4):
            M = i + 1
            a = (self._dfac([n], 2)[0] + k * self._dfac([M], 2)[0])
            b = stats.mean(self._dfac(x, M)) / self._dfac([n - 2], M - 2)[0]
            c = k * self._dfac([M], 2)[0] * m[i]
            m[i] = (b / a) + c
        return m

    def _betaparameters(self, x: list, n: int, k: float, model: int = 4, l: float = 0, u: float = 1):
        """
        Estimate the parameters of a two- or four-parameter beta distribution for the true-score distribution.

        Parameters
        ----------
        x : list
            List of values representing final test scores.
        n : int
            Test length. For the Livingston and Lewis approach, this is the effective test length.
            For the Hanson and Brennan approach, this is the actual test length.
        k : float
            Lord's k parameter, used for adjusting the distribution.
        model : int, optional
            Specifies the model to use for parameter estimation:
            - 2: Two-parameter beta distribution.
            - 4: Four-parameter beta distribution. Default is 4.
        l : float, optional
            Lower bound of the beta distribution. Used only if `model` is 2. Default is 0.
        u : float, optional
            Upper bound of the beta distribution. Used only if `model` is 2. Default is 1.

        Returns
        -------
        dict
            A dictionary containing the estimated parameters of the beta distribution:
            - 'alpha': The first shape parameter (alpha).
            - 'beta': The second shape parameter (beta).
            - 'l': The lower bound (applicable to both models, default is 0 for two-parameter model).
            - 'u': The upper bound (applicable to both models, default is 1 for two-parameter model).

        Examples
        --------
        >>> _betaparameters([5, 6, 7, 8], n=10, k=0.2, model=4)
        {'alpha': 2.5, 'beta': 3.0, 'l': 0.0, 'u': 1.0}

        >>> _betaparameters([5, 6, 7, 8], n=10, k=0.2, model=2, l=0, u=1)
        {'alpha': 1.8, 'beta': 2.2, 'l': 0, 'u': 1}
        """
        m = self._tsm(x, n, k)
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


## TESTS: ###
# Setting the seed
np.random.seed(123456)
from support_functions.betafunctions import cronbachs_alpha, rbeta4p

# Define the parameters for the beta distribution
a, b = 6, 4
#The first two parameters are for the location and scale parameters respectively
p_success = rbeta4p(1000, 6, 4, .15, .85)

# Preallocate a matrix of zeros with 1000 rows and 20 columns
rawdata = np.zeros((1000, 100))

# Loop over the columns
for i in range(1000):
    for j in range(100):
        rawdata[i, j] = np.random.binomial(1, p_success[i], 1)
sumscores = list(np.sum(rawdata, axis = 1))
meanscores = np.mean(rawdata, axis = 1)
output = bbclassify(sumscores, cronbachs_alpha(rawdata), 0, 100, [50, 75], method = "ll")
output.modelfit().Modelfit
output.accuracy().Accuracy
output.consistency().Consistency
print(output.accuracy().Accuracy)
print(o)
#output.Accuracy().caprint()
#output.Consistency().caprint()
#print(f"Accuracy: {output.accuracy}.\nConsistency: {output.consistency}")