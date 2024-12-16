from warnings import warn
from typing import Union
from scipy.integrate import quad
from scipy.stats import binom, chi2
import scipy.special
import statistics as stats
import math
import numpy as np
import pandas as pd

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
        
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        
        >>> bb_ll = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score= 0, max_score = 100, cut_scores = [50, 75], method = "ll")
        >>> print(bb_ll.Parameters)
        {'alpha': 3.877386083988672, 'beta': 3.9727308136649238, 'l': 0.2681195709670965, 'u': 0.8706384086828665, 'etl': 99.96892140618861, 'etl rounded': 100, 'lords_k': 0}
        
        >>> bb_ll.modelfit()
        >>> print([bb_ll.Modelfit_chi_squared, bb_ll.Modelfit_degrees_of_freedom, bb_ll.Modelfit_p_value])
        [41.57098815343608, 46, 0.6581136565975114]

        >>> bb_ll.accuracy()
        0.8438848734448846
        >>> print(bb_ll.Accuracy)
        0.8438848734448846

        >>> bb_ll.consistency()
        0.7811757067805466
        >>> print(bb_ll.Consistency)
        0.7811757067805466

        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score= 0, max_score = 100, cut_scores = [50, 75], method = "hb")
        >>> print(bb_hb.Parameters)
        {'alpha': 3.878383371886145, 'beta': 3.974443224813199, 'l': 0.2680848232389114, 'u': 0.8707270089303472, 'lords_k': -0.015544127802040899}
        >>> bb_hb.modelfit()
        >>> print([bb_hb.Modelfit_chi_squared, bb_hb.Modelfit_degrees_of_freedom, bb_hb.Modelfit_p_value])
        [41.568240407567785, 46, 0.6582260821176256]
        >>> bb_hb.accuracy()
        0.8440449341145039
        >>> print(bb_hb.Accuracy)
        0.8440449341145039
        >>> bb_hb.consistency()
        0.7814787747625861
        >>> print(bb_hb.Consistency)
        0.7814787747625861
        """
        # Input validation
        if not isinstance(data, (list, tuple)):
            raise TypeError("data must be a list or tuple of scores.")
        if not all(isinstance(d, (int, float)) for d in data):
            raise ValueError("All elements in data must be integers or floats.")
        if True in [True if (i < min_score or i > max_score) else False for i in data]:
            raise ValueError("Values in data can not be smaller than the specified minimum or greater than the specified maximum possible score.")
        if not isinstance(reliability, float) or not (0 <= reliability <= 1):
            raise ValueError("reliability must be a float between 0 and 1.")
        if not isinstance(min_score, (int, float)) or not isinstance(max_score, (int, float)):
            raise TypeError("min_score and max_score must be numeric.")
        if min_score >= max_score:
            raise ValueError("min_score must be less than max_score.")
        if not isinstance(cut_scores, list) or not all(isinstance(cs, (int, float)) for cs in cut_scores):
            raise TypeError("cut_scores must be a list of numeric values.")
        if not all(min_score < cs < max_score for cs in cut_scores):
            raise ValueError("All cut_scores must be between min_score and max_score.")
        if method not in ["ll", "hb"]:
            raise ValueError("method must be 'll' or 'hb'.")
        if model not in [2, 4]:
            raise ValueError("model must be 2 or 4.")
        if not (0 <= l < u <= 1):
            raise ValueError("l and u must satisfy 0 <= l < u <= 1.")
        if not isinstance(failsafe, bool):
            raise TypeError("failsafe must be a boolean.")

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

        self.Modelfit_chi_squared = "Model fit not yet estimated. Call .modelfit() to estimate model fit."
        self.Modelfit_degrees_of_freedom = "Model fit not yet estimated. Call .modelfit() to estimate model fit."
        self.Modelfit_p_value = "Model fit not yet estimated. Call .modelfit() to estimate model fit."
        self.Accuracy = "Accuracy not yet estimated. Call .accuracy() to estimate model fit."
        self.Consistency = "Consistency not yet estimated, Call .consistency() to estiamate model fit."

        if isinstance(self.data, dict): # Parameters do not have to be estimated if a dict of parameter values is supplied.
            self.Parameters = self.data
            if self.method == "ll":
                self.N = self.Parameters["etl"]
            else:
                self.N = self.Parameters["atl"]
        else: # If raw data is supplied, estimate parameters from the data.
            if self.method == "ll": # For the Livingston and Lewis method:
                self.effective_test_length = self._calculate_etl(stats.mean(self.data), stats.variance(self.data), float(self.reliability), self.min_score, self.max_score)
                self.N = round(self.effective_test_length)
                self.K = 0
                self.Parameters = self._betaparameters(self.data, self.N, 0, self.model, self.l, self.u)
                self.Parameters["etl"] = self.effective_test_length
                self.Parameters["etl rounded"] = self.N
                self.Parameters["lords_k"] = 0
            else: # For the Hanson and Brennan method:
                self.N = self.max_score
                self.K = float(self._calculate_lords_k(stats.mean(self.data), stats.variance(self.data), self.reliability, self.N))
                self.Parameters = self._betaparameters(self.data, self.N, self.K, self.model, self.l, self.u)
                self.Parameters["lords_k"] = self.K
            # If a four-parameter fitting procedure produced invalid location parameter estimates, and the failsafe was specified to
            # engage in such a circumstance, fit a two-parameter model instead with the location-parameters specified by the user.
            if (self.failsafe == True and self.model == 4) and (self.Parameters["l"] < 0 or self.Parameters["u"] > 1):
                warn(f"Failsafe triggered. True-score fitting procedure produced impermissible parameter values (l = {self.Parameters['l']}, u = {self.Parameters['u']}).")
                self.Parameters = self._betaparameters(self.data, self.N, self.K, 2, self.l, self.u)
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
                    self._bbintegrate1_1(self.Parameters["alpha"], self.Parameters["beta"], self.Parameters["l"], self.Parameters["u"],
                            self.choose_values[i], self.N, i, self.Parameters["lords_k"], 0, 1, self.method)[0] * n_respondents 
                            for i in range(self.N + 1)
                    ]
        # Make a list of the actual observed-score frequency-distribution.
        # If method == "ll", rescale observed-scores onto the effective test-length scale, and round to integer for count.
        if self.method == "ll": 
            observed = [round(((i - self.min_score) / (self.max_score - self.min_score)) * self.N) for i in self.data]
            observed = [observed.count(i) for i in range(self.N + 1)]
        else: # Otherwise, simply count frequencies of each score from 0 to N items.
            observed = [self.data.count(i) for i in range(self.N + 1)]

        # Collapse -- first left-to-right, then right-to-left --- cells with expected values less than 1.
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
        self.Modelfit_p_value = 1 - float(chi2.cdf(self.Modelfit_chi_squared, self.Modelfit_degrees_of_freedom))
    
    # Function for estimating classification accuracy.
    def accuracy(self):
        # Initialize a matrix of proportions of individuals with true-scores within each category producing specific observed-scores.
        confmat = np.zeros((self.N + 1, len(self.cut_scores) - 1))
        for i in range(len(self.cut_scores) - 1):
            for j in range(self.N + 1):
                confmat[j, i] = self._bbintegrate1_1(self.Parameters["alpha"], self.Parameters["beta"], self.Parameters["l"], self.Parameters["u"], 
                                               self.choose_values[j], self.N, j, self.Parameters["lords_k"], self.cut_truescores[i], self.cut_truescores[i + 1], self.method)[0]
        # Initialize the confusion matrix collapsing (summing) the values in "confmat" into ranges of observed-values corresponding to each category.
        self.confusionmatrix = np.zeros((len(self.cut_scores) - 1, len(self.cut_scores) - 1))
        for i in range(len(self.cut_scores) - 1):
            for j in range(len(self.cut_scores) - 1):
                if i != len(self.cut_scores) - 2:
                    self.confusionmatrix[i, j] = sum(confmat[self.cut_scores[i]:self.cut_scores[i + 1], j]) 
                else:
                    self.confusionmatrix[i, j] = sum(confmat[self.cut_scores[i]:, j])
        # Compute overall accuracy by summing the values in the diagonal of the confusion matrix.
        self.Accuracy = float(sum([self.confusionmatrix[i, i] for i in range(len(self.cut_scores) - 1)]))
        return self.Accuracy

    # Function for estimating classification consistency.
    def consistency(self):
        # Initialize confusion matrix where proportions of consecutive item-scores are calculated.
        consmat = np.zeros((self.N + 1, self.N + 1))
        # Since the consistency matrix is a symmetric matrix only the lower triangle is computed for performance considerations.
        for i in range(self.N + 1):
            for j in range(self.N + 1):
                if i <= j:
                    consmat[i, j] = self._bbintegrate2_1(self.Parameters["alpha"], self.Parameters["beta"], self.Parameters["l"], self.Parameters["u"], 
                                                   self.choose_values[i], self.choose_values[j], self.N, i, j, self.Parameters["lords_k"], 0, 1, self.method)[0]
        lower_triangle = np.tril_indices(consmat.shape[0], 0)
        # Transpose the lower triangle and insert into the upper triangle to fill out the matrix.
        consmat[lower_triangle] = consmat.T[lower_triangle]
        # Initialize confusion matrix for categories.
        self.consistencymatrix = np.zeros((len(self.cut_scores) - 1, len(self.cut_scores) - 1))
        # Collapse (sum) values in ranges to produce the consistency matrix for categories.
        # There are 9 different cases.
        # Corners (4 cases)
        # Sides (4 cases)
        # Center (1 case)
        for i in range(len(self.cut_scores) - 1):
            for j in range(len(self.cut_scores) - 1):
                if i == 0 and j == 0: 
                    # Top-left corner.
                    self.consistencymatrix[i, j] = sum(sum(consmat[0:self.cut_scores[i + 1], 0:self.cut_scores[j + 1]]))
                elif i == 0 and (j != 0 and j != len(self.cut_scores) - 2): 
                    # Top-side.
                    self.consistencymatrix[i, j] = sum(sum(consmat[0:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1]]))
                elif i == 0  and j == len(self.cut_scores) - 2: 
                    # Top-right corner.
                    self.consistencymatrix[i, j] = sum(sum(consmat[0:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1] + 1]))
                elif (i != 0 and i != len(self.cut_scores) - 2) and j == 0:
                    # Left-side.
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1], 0:self.cut_scores[j + 1]]))
                elif (i != 0 and i != len(self.cut_scores) - 2) and (j != 0 and j != len(self.cut_scores) - 2):
                    # Center
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1]]))
                elif (i != 0 and i != len(self.cut_scores) - 2) and j == len(self.cut_scores) - 2:
                    # Right-side
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1], self.cut_scores[j]:self.cut_scores[j + 1] + 1]))
                elif i == len(self.cut_scores) - 2 and j == 0:
                    # Bottom-left corner.
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1] + 1, 0:self.cut_scores[j + 1]]))
                elif i == len(self.cut_scores) - 2 and (j != 0 and j != len(self.cut_scores) - 2):
                    # Bottom-side
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1] + 1, self.cut_scores[j]:self.cut_scores[j + 1]]))
                else:
                    # Bottom-right corner
                    self.consistencymatrix[i, j] = sum(sum(consmat[self.cut_scores[i]:self.cut_scores[i + 1] + 1, self.cut_scores[j]:self.cut_scores[j + 1] + 1]))
        # Compute overall consistency by summing the values in the diagonal of the consistency matrix.
        self.Consistency = float(sum([self.consistencymatrix[i, i] for i in range(len(self.cut_scores) - 1)]))
        return self.Consistency
        
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_ll = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score= 0, max_score = 100, cut_scores = [50, 75], method = "ll")
        >>> print(bb_ll._calculate_etl(mean = stats.mean(sumscores), var = stats.variance(sumscores), reliability = reliability(rawdata).alpha(), min = 0, max = 100))
        99.96892140618861
        """
        if not isinstance(mean, (int, float)) or not isinstance(var, (int, float)):
            raise TypeError("mean and var must be numeric.")
        if not isinstance(reliability, float) or not (0 <= reliability <= 1):
            raise ValueError("reliability must be a float between 0 and 1.")
        if not isinstance(min, (int, float)) or not isinstance(max, (int, float)):
            raise TypeError("min and max must be numeric.")
        if min >= max:
            raise ValueError("min must be less than max.")
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score= 0, max_score = 100, cut_scores = [50, 75], method = "hb")
        >>> print(bb_hb._calculate_lords_k(mean = stats.mean(sumscores), var = stats.variance(sumscores), reliability = reliability(rawdata).alpha(), length = 100))
        -0.015544127802040899
        """
        vare = var * (1 - reliability)
        num = length * ((length - 1) * (var - vare) - length * var + mean * (length - mean))
        den = 2 * (mean * (length - mean) - (var - vare))
        return num / den

    def _rbeta4p(self, n: int, a: float, b: float, l: float = 0, u: float = 1) -> np.array:
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
            Array of length 'n' containing random values drawn from the four-parameter beta distribution.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bbclassify(sumscores, reliability(rawdata).alpha(), 0, 100, [50])._rbeta4p(5, 6, 4, .25, .75)
        array([0.53467969, 0.70218754, 0.45730315, 0.5583427 , 0.59158903])
        """
        return np.random.beta(a, b, n) * (u - l) + l

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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score= 0, max_score = 100, cut_scores = [50, 75], method = "hb")
        >>> print(bb_hb._dbeta4p(x = 0.5, a = 6, b = 4, l = 0.15, u = 0.85))
        2.8124999999999996
        """
        if x < l or x > u:
            return 0
        else:
            return (1 / scipy.special.beta(a, b)) * (((x - l)**(a - 1) * (u - x)**(b - 1)) / (u - l)**(a + b - 1))

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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score= 0, max_score = 100, cut_scores = [50, 75], method = "hb")
        >>> bb_hb._dcbinom(p = 0.5, N = 10, n = 5, k = 2.0)
        0.3007812500000001
        """
        a = binom.pmf(n, N, p)
        b = binom.pmf(n, N - 2, p)
        c = binom.pmf(n - 1, N - 2, p)
        d = binom.pmf(n - 2, N - 2, p)
        e = k * p * (1 - p)
        return float(a - e * (b - 2*c + d))

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
        
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
        """
        a = x[0]*(p**n)*(1 - p)**(N - n)
        if method != "ll":
            b = x[1]*(p**n)*(1 - p)**(N - n)
            c = x[2]*(p**n)*(1 - p)**(N - n)
            d = x[3]*(p**n)*(1 - p)**(N - n)
            e = k * p * (1 - p)
            return a - e * (b - 2*c + d)
        return a

    def _da_factorial(self, x: int) -> int:
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50])
        >>> print(bb._da_factorial(5))
        120
        >>> print(bb._da_factorial(0))
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50])
        >>> print(bb._choose(10, 5))
        252
        """
        return int(self._da_factorial(N) / (self._da_factorial(n) * self._da_factorial(N - n)))

    def _choose_functions(self, N, n) -> tuple:
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50])
        >>> print(bb._choose_functions(10, 5))
        (252, 56, 70, 56)
        """        
        a = self._choose(N, n)
        b = self._choose(N - 2, n)
        c = self._choose(N - 2, n - 1)
        d = self._choose(N - 2, n - 2)
        return (a, b, c, d)

    def _bbintegrate1(self, a: float, b: float, l: float, u: float, N: int, n: int, k: float, lower: float, upper: float, method: str = "ll") -> tuple:
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
        tuple
            The computed area under the curve for the beta-binomial distribution over the specified range.
            - [0] : The area under the curve.
            - [1] : Error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50])
        >>> print(bb_hb._bbintegrate1(6, 4, 0.15, 0.85, 10, 5, 1, 0, 1, method = 'll'))
        (0.18771821236360686, 1.0678646219550548e-08)
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
        tuple
            The computed area under the curve for the beta-binomial distribution over the specified range.
            - [0] : The area under the curve.
            - [1] : Error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
        >>> cf = bb_hb._choose_functions(10, 5)
        >>> print(bb_hb._bbintegrate1_1(6, 4, 0.15, 0.85, cf, 10, 5, 1, 0, 1, method = 'll'))
        (0.18771821236360692, 1.0678646219447462e-08)
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
        tuple
            The computed area under the curve for the beta-binomial distribution over the specified range.
            - [0] : The area under the curve.
            - [1] : Error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
        >>> print(bb_hb._bbintegrate2(6, 4, 0.15, 0.85, 10, 5, 5, 1, 0, 1, method = 'll'))
        (0.03843435178500844, 4.336562943457941e-10)
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
        tuple
            The computed area under the curve for the beta-binomial distribution over the specified range.
            - [0] : The area under the curve.
            - [1] : Error estimate.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
        >>> cf = bb_hb._choose_functions(10, 5)
        >>> print(bb_hb._bbintegrate2_1(6, 4, 0.15, 0.85, cf, cf, 10, 5, 5, 1, 0, 1, method = 'll'))
        (0.03843435178500849, 4.336562889266626e-10)
        """
        if method != "ll":
            def f(x, a, b, l, u, c1, c2, N, n1, n2, k):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom2(c1, x, N, n1, k, method) * self._dcbinom2(c2, x, N, n2, k, method)
            return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2, k))
        else:
            def f(x, a, b, l, u, c1, c2, N, n1, n2):
                return self._dbeta4p(x, a, b, l, u) * self._dcbinom2(c1, x, N, n1, k, method) * self._dcbinom2(c2, x, N, n2, k, method)
            return quad(f, lower, upper, args = (a, b, l, u, c1, c2, N, n1, n2))

    def _dfac(self, x: list, r = int) -> list:
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
        >>> bb_hb._dfac([5, 6, 7], 3)
        [60, 120, 210]

        >>> bb_hb._dfac([4, 3], 1)
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

    def _tsm(self, x: list, n: int, k: float) -> list:
        """
        Calculate the first four moments of the true-score distribution.

        Parameters
        ----------
        x : list
            List of values representing final test scores.
        n : int
            - Effective test length for the Livingston and Lewis approach.
            - Actual test length for the Hanson and Brennan approach.
        k : float
            Lord's k parameter, used for moment estimation under the Hanson and Brennan approach.

        Returns
        -------
        list
            A list containing the first four moments of the true-score distribution, in order:
            - [0] : The mean.
            - [1] : The variance.
            - [2] : The skewness.
            - [3] : The kurtosis.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
        >>> print(bb_hb._tsm(sumscores, 100, bb_hb.Parameters['lords_k']))
        [0.56572, 0.33029356244956504, 0.19847130696480725, 0.12240805759909551]
        """
        m = [stats.mean(x) / n, 0, 0, 0]
        for i in range(1, 4):
            M = i + 1
            a = (self._dfac([n], 2)[0] + k * self._dfac([M], 2)[0])
            b = stats.mean(self._dfac(x, M)) / self._dfac([n - 2], M - 2)[0]
            c = k * self._dfac([M], 2)[0] * m[i]
            m[i] = (b / a) + c
        return m

    def _betaparameters(self, x: list, n: int, k: float, model: int = 4, l: float = 0, u: float = 1) -> dict:
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
        >>> bb_hb = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
        >>> print(bb_hb.Parameters)
        {'alpha': 3.878383371886145, 'beta': 3.974443224813199, 'l': 0.2680848232389114, 'u': 0.8707270089303472, 'lords_k': -0.015544127802040899}
        >>> bb_ll = bbclassify(data = sumscores, reliability = reliability(rawdata).alpha(), min_score= 0, max_score = 100, cut_scores = [50, 75], model = 2, l = 0.25, u = 0.85, method = "ll")
        >>> print(bb_ll.Parameters)
        {'alpha': 4.079875519795486, 'beta': 3.673593731051123, 'l': 0.25, 'u': 0.85, 'etl': 99.96892140618861, 'etl rounded': 100, 'lords_k': 0}
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

class reliability():
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the reliability object, calculating the covariance matrix from the raw data.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 5, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> rel = reliability(rawdata)
        >>> print(rel.covariance_matrix)
        [[ 0.24689157  0.00293976  0.01241767 -0.02012851 -0.00817671]
         [ 0.00293976  0.24869076  0.00178313  0.01653012  0.02549398]
         [ 0.01241767  0.00178313  0.24322892  0.03934137  0.0055743 ]
         [-0.02012851  0.01653012  0.03934137  0.2499759   0.05113253]
         [-0.00817671  0.02549398  0.0055743   0.05113253  0.241751  ]]
        """
        self.data = data
        self.covariance_matrix = np.array(self.data.dropna().cov())

    def alpha(self) -> float:
        """
        Calculate the Cronbach's Alpha reliability coefficient.

        Returns
        -------
        float
            A float value representing Cronbach's Alpha.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)])
        >>> rel = reliability(rawdata)
        >>> alpha = rel.alpha()
        >>> print(round(alpha, 2))
        0.81
        """
        n = self.covariance_matrix.shape[0]
        self.Alpha = (n / (n - 1)) * (1 - (sum(np.diag(self.covariance_matrix)) / sum(sum(self.covariance_matrix))))
        return self.Alpha
    
    def omega(self) -> float:
        """
        Calculate the McDonald's Omega reliability coefficient using the unweighed least squares method.

        Returns
        -------
        float
            A float value representing McDonald's Omega.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(1234)
        >>> N_resp, N_items, alpha, beta, l, u = 250, 10, 6, 4, .15, .85
        >>> p_success = np.random.beta(alpha, beta, N_resp) * (u - l) + l
        >>> rawdata = pd.DataFrame([np.random.binomial(5, p_success[i], N_items) for i in range(N_resp)])
        >>> rel = reliability(rawdata)
        >>> omega = rel.omega()
        >>> print(round(omega, 2))
        0.72
        """
        variance_list = np.diag(self.covariance_matrix)
        self.factor_loadings = []
        for _ in range(len(variance_list)):
            factor_loading = []
            covariance_list = [[float(self.covariance_matrix[i + j + 1, i]) for j in range(len(self.covariance_matrix[i:, i]) - 1)] for i in range(len(self.covariance_matrix[0]) - 1)]
            for i in range(len(covariance_list[0]) - 1):
                for j in range(len(covariance_list[i + 1])):
                    # If a covariance is exactly 0, consider it a rounding error and add 0.0001.
                    if covariance_list[i + 1][j] < 0:
                        raise ValueError("Covariance matrix contains negative values. Coefficient Omega will not be an appropriate measure of reliability.")
                    if abs(covariance_list[i + 1][j]) == 0: covariance_list[i + 1][j] += .00001
                    value = (covariance_list[0][i] * covariance_list[0][i + j + 1])  / covariance_list[i + 1][j]
                    factor_loading.append(value**.5)
            self.factor_loadings.append(stats.mean(factor_loading))
            self.covariance_matrix = np.vstack([self.covariance_matrix, self.covariance_matrix[[0], :]])
            self.covariance_matrix = np.hstack([self.covariance_matrix, self.covariance_matrix[:, [0]]])
            self.covariance_matrix = self.covariance_matrix[1:, 1:]
        squared_factor_loadings = [i**2 for i in self.factor_loadings]
        factor_loadings_squared = sum(self.factor_loadings)**2
        def omega_gfi():
            implied_matrix = np.zeros((self.covariance_matrix.shape[0], self.covariance_matrix.shape[0]), float)
            for i in range(self.covariance_matrix.shape[0]):
                for j in range(self.covariance_matrix.shape[0]):
                    if i != j:
                        implied_matrix[i, j] = self.factor_loadings[i] * self.factor_loadings[j]
                    else:
                        implied_matrix[i, j] = self.covariance_matrix[i, j]
            self.Omega_GFI = 1 - (((self.covariance_matrix - implied_matrix).mean())**2 / self.covariance_matrix.mean())
            return self.Omega_GFI
        self.omega_gfi = omega_gfi
        self.Omega = factor_loadings_squared / (sum([variance_list[i] - squared_factor_loadings[i] for i in range(len(variance_list))]) + factor_loadings_squared)
        return self.Omega

if __name__ == "__main__":
    import doctest
    doctest.testmod()
