from support_functions.betafunctions import *
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
                self.N = pars["etl"]
            else:
                self.N = pars["atl"]
        else:
            if method == "ll":
                self.effective_test_length = etl(stats.mean(self.data), stats.variance(self.data), self.reliability, self.min_score, self.max_score)
                self.N = round(self.effective_test_length)
                self.parameters = betaparameters(self.data, self.N, 0, model, l, u)
                if (self.failsafe == True and self.model == 4) and (self.l < 0 or self.u > 1):
                    self.parameters = betaparameters(self.data, self.N, 0, 2, self.l, self.u)
                self.parameters["etl"] = self.effective_test_length
                self.parameters["etl rounded"] = self.N
                self.parameters["lords_k"] = 0
            else:
                self.N = self.max_score
                self.K = k(stats.mean(self.data), stats.variance(self.data), self.reliability, self.N)
                self.parameters = betaparameters(self.data, self.N, self.K, self.model, self.l, self.u)
                if (self.failsafe == True and self.model == 4) and (self.l < 0 or self.u > 1):
                    pars = betaparameters(self.data, self.max_score, self.N, 2, self.l, self.u)
                self.parameters["lords_k"] = self.K
        
        self.choose_values = [choose_functions(self.N, i) for i in range(self.N + 1)]
    
    def modelfit(self):
        if isinstance(self.data, dict):
            raise ValueError("Model fit testing requires observed test-scores as data input.")
        n_respondents = len(self.data)
        if self.method != "ll":
            expected = [
                bbintegrate1_2(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"],
                               self.choose_values[i], self.N, i, self.parameters["lords_k"], self.parameters["l"], self.parameters["u"], self.method)[0] * n_respondents
                               for i in range(self.N + 1)
                               ]
            observed = [self.data.count(i) for i in range(self.N + 1)]
            for _ in range(2):
                length = len(expected)
                for j in range(length):
                    if expected[j + 1] < 1:
                        expected[j + 1] += expected[j]
                        observed[j + 1] += observed[j]
                        expected[j], observed[j] = None, None
                    else:
                        break
                observed = [observed[j] for j in range(length) if expected[j] is not None][::-1]
                expected = [j for j in expected if j is not None][::-1]
            normalize = sum(observed) / sum(expected)
            expected = [j*normalize for j in expected if j is not None]
        else:
            None
        #fig, ax = plt.subplots()
        #plt.bar([str(i) for i in range(len(observed))], observed)
        #plt.show()
        print(f"Total: {len(self.data)}")
        print(f"EXPECTED:: Length: {len(expected)}. Min: {min(expected)}. Max: {max(expected)}. Sum: {sum(expected)}. 1st value: {expected[0]}. Last value: {expected[-1]}.")
        print(f"OBSERVED:: Length: {len(observed)}. Min: {min(observed)}. Max: {max(observed)}. Sum: {sum(observed)}. 1st value: {observed[0]}. Last value: {observed[-1]}.")
        print(expected.index(min(expected)))
    
    def accuracy(self):
        confmat = np.zeros((self.N + 1, len(self.cut_scores) - 1))
        for i in range(len(self.cut_scores) - 1):
            for j in range(self.N + 1):
                confmat[j, i] = bbintegrate1_2(self.parameters["alpha"], self.parameters["beta"], self.parameters["l"], self.parameters["u"], 
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
output = bbclassify(sumscores, cronbachs_alpha(rawdata), 0, 100, [50, 75], method = "hb").modelfit()
#output.Accuracy().caprint()
#output.Consistency().caprint()
#print(f"Accuracy: {output.accuracy}.\nConsistency: {output.consistency}")