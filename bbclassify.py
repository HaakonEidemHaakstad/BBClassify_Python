from support_functions.betafunctions import *
from warnings import warn
import matplotlib.pyplot as plt
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

        self.Accuracy = None
        self.Consistency = None

        if isinstance(self.data, dict):
            self.parameters = self.data
            if method == "ll":
                self.N = self.parameters["etl"]
            else:
                self.N = self.parameters["atl"]
        else:
            if method == "ll":
                self.effective_test_length = etl(stats.mean(self.data), stats.variance(self.data), self.reliability, self.min_score, self.max_score)
                self.N = round(self.effective_test_length)
                self.K = 0
                self.parameters = betaparameters(self.data, self.N, 0, self.model, self.l, self.u)
                self.parameters["etl"] = self.effective_test_length
                self.parameters["etl rounded"] = self.N
                self.parameters["lords_k"] = 0
            else:
                self.N = self.max_score
                self.K = k(stats.mean(self.data), stats.variance(self.data), self.reliability, self.N)
                self.parameters = betaparameters(self.data, self.N, self.K, self.model, self.l, self.u)
                self.parameters["lords_k"] = self.K
            if (self.failsafe == True and self.model == 4) and (self.parameters["l"] < 0 or self.parameters["u"] > 1):
                warn(f"Failsafe triggered. True-score fitting procedure produced impermissible parameter values (l = {self.parameters['l']}, u = {self.parameters['u']}).")
                self.parameters = betaparameters(self.data, self.N, self.K, 2, self.l, self.u)
                self.model = 2

        self.choose_values = [choose_functions(self.N, i) for i in range(self.N + 1)]
    
    # Function for testing model fit.
    def modelfit(self):
        # If the input to the data argument is a dictionary of model-parameters, throw a value error.
        if isinstance(self.data, dict):
            raise ValueError("Model fit testing requires observed test-scores as data input.")
        n_respondents = len(self.data)
        # Make a list of the model-implied (expected) frequency-distribution.
        expected = [
                    bbintegrate1_1(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"],
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
        self.modelfit_chi_squared = sum([(observed[i] - expected[i])**2 / expected[i] for i in range(len(expected))])
        self.modelfit_degrees_of_freedom = len(expected) - self.model
        self.modelfit_p_value = 1 - chi2.cdf(self.modelfit_chi_squared, self.modelfit_degrees_of_freedom)

        print(self.modelfit_p_value)
        print(f"Total: {len(self.data)}")
        print(f"EXPECTED:: Length: {len(expected)}. Min: {min(expected)}. Max: {max(expected)}. Sum: {int(sum(expected))}. 1st value: {expected[0]}. Last value: {expected[-1]}.")
        print(f"OBSERVED:: Length: {len(observed)}. Min: {min(observed)}. Max: {max(observed)}. Sum: {sum(observed)}. 1st value: {observed[0]}. Last value: {observed[-1]}.")
        print(expected.index(min(expected)))
        print(f"Chi-squared: {self.modelfit_chi_squared}.")
    
    # Function for estimating classification accuracy.
    def accuracy(self):
        confmat = np.zeros((self.N + 1, len(self.cut_scores) - 1))
        for i in range(len(self.cut_scores) - 1):
            for j in range(self.N + 1):
                confmat[j, i] = bbintegrate1_1(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"], 
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
                    consmat[i, j] = bbintegrate2_1(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"], 
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
        
    def caprint(self):
        if self.accuracy != None:
            print(f"Overall accuracy: {self.accuracy}")
        if self.consistency != None:
            print(f"Overall consistency: {self.consistency}")

## TESTS: ###
# Setting the seed
np.random.seed(123456)

# Define the parameters for the beta distribution
a, b = 6, 4
#The first two parameters are for the location and scale parameters respectively
p_success = rbeta4p(10000, 6, 4, .15, .85)

# Preallocate a matrix of zeros with 1000 rows and 20 columns
rawdata = np.zeros((10000, 100))

# Loop over the columns
for i in range(10000):
    for j in range(100):
        rawdata[i, j] = np.random.binomial(1, p_success[i], 1)
sumscores = list(np.sum(rawdata, axis = 1))
meanscores = np.mean(rawdata, axis = 1)
output = bbclassify(sumscores, cronbachs_alpha(rawdata), 0, 100, [50, 75], method = "ll").modelfit()
#output.Accuracy().caprint()
#output.Consistency().caprint()
#print(f"Accuracy: {output.accuracy}.\nConsistency: {output.consistency}")