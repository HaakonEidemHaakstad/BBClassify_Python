from support_functions.betafunctions import *

## Function for estimating accuracy and consistency from beta-binomial models.
# x = vector of values representing test-scores, or a list of model parameters.
# reliability = the reliability coefficient of the test-scores.
# min = the minimum possible score to attain on the test (only necessary for 
#       the Livingston and Lewis approach).
# max = for the Livingston and Lewis approach, the maximum possible score to 
#       attain on the test. For the Hanson and Brennan approach, the actual
#       test length (number of items).
# model = how many parameters of the true-score distribution that is to be
#       estimated (4 or 2). Default is 4.
# l = the lower-bound location parameter for the two-parameter distribution.
# u = the lower-bound location parameter for the two-parameter distribution.
# failsafe = whether the function should automatically revert to a two-
#       parameter solution if the four-parameter fitting procedure produces
#       impermissible location-parameter estimates.
# method = string specifying whether it is the Livingston and Lewis or the 
#       Hanson and Brennan approach is to be employed. Default is "ll" 
#       (Livingston and Lewis). Any other value passed means the Hanson and 
#       Brennan approach.
# output = list of strings which state what analyses to do and include in the 
#       output.
# print_output = Whether to print the output to the consoel as it is estimated.
def cac(x, reliability: float, min: float, max: float, cut: float, model: int = 4, l: float = 0, u: float = 1, failsafe: bool = False, method: str = "ll", output: list[str] = ["parameters", "accuracy", "consistency"], print_output = True):
    out = {}
    cut = [min] + cut + [max]
    tcut = list(cut)
    for i in range(len(cut)):
        tcut[i] = (tcut[i] - min) / (max - min)
    if isinstance(x, dict):
        pars = x
        if method == "ll":
            N = pars["etl"]
        else:
            N = pars["atl"]
    else:
        if method == "ll":
            Nnotrounded = etl(stats.mean(x), stats.variance(x), reliability, min, max)
            N = round(Nnotrounded)
            pars = betaparameters(x, N, 0, model, l, u)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, N, 0, 2, l, u)
            pars["etl"] = Nnotrounded
            pars["etl rounded"] = N
            pars["lords_k"] = 0
            for i in range(len(cut)):
                cut[i] = tcut[i] * N
                cut[i] = round(cut[i])
        else:
            N = max 
            K = k(stats.mean(x), stats.variance(x), reliability, N)
            pars = betaparameters(x, N, K, model, l, u)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, max, N, 2, l, u)
            pars["lords_k"] = K
    if "parameters" in output:
        out["parameters"] = pars
        if print_output:
            print("Model parameter estimates:")
            print(out["parameters"])
            print("")

    if "accuracy" in output:
        print("Estimating accuracy...\n")
        confmat = np.zeros((N + 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(N + 1):
                confmat[j, i] = bbintegrate1(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, j, pars["lords_k"], tcut[i], tcut[i + 1], method)[0]
        confusionmatrix = np.zeros((len(cut) - 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(len(cut) - 1):
                if i != len(cut) - 2:
                    confusionmatrix[i, j] = sum(confmat[cut[i]:cut[i + 1], j])
                else:
                    confusionmatrix[i, j] = sum(confmat[cut[i]:, j])
        accuracy = []
        for i in range(len(cut) - 1):
            accuracy = accuracy + [confusionmatrix[i, i]]
        accuracy = sum(accuracy)
        out["Confusion matrix"] = pd.DataFrame(confusionmatrix)
        out["Overall accuracy"] = accuracy
        if print_output:
            print("Confusion matrix:")
            print(out["Confusion matrix"])
            print("")
            print("Overall accuracy:")
            print(out["Overall accuracy"])
            print("")
    
    if "consistency" in output:
        print("Estimating consistency...\n")
        consmat = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                if i <= j:
                    consmat[i, j] = bbintegrate2(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, i, j, pars["lords_k"], 0, 1, method)[0]
        lower_triangle = np.tril_indices(consmat.shape[0], 0)
        consmat[lower_triangle] = consmat.T[lower_triangle]
        consistencymatrix = np.zeros((len(cut) - 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(len(cut) - 1):
                if i == 0 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], 0:cut[j + 1]]))
                if i == 0 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1]]))
                if i == 0  and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                if (i != 0 and i != len(cut) - 2) and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], 0:cut[j + 1]]))
                if (i != 0 and i != len(cut) - 2) and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1]]))
                if (i != 0 and i != len(cut) - 2) and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                if i == len(cut) - 2 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, 0:cut[j + 1]]))
                if i == len(cut) - 2 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1]]))
                if i == len(cut) - 2 and j == len(cut) - 2:
                        consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1] + 1]))
            consistency = []
            for i in range(len(cut) - 1):
                consistency = consistency + [consistencymatrix[i, i]]
            consistency = sum(consistency)
        out["Consistency matrix"] = pd.DataFrame(consistencymatrix)
        out["Overall consistency"] = consistency
        if print_output:
            print("Consistency matrix:")
            print(out["Consistency matrix"])
            print("")
            print("Overall consistency:")
            print(out["Overall consistency"])
    return out

def cac2(x, reliability: float, min: float, max: float, cut: float, model: int = 4, l: float = 0, u: float = 1, failsafe: bool = False, method: str = "ll", output: list[str] = ["parameters", "accuracy", "consistency"], print_output = True):
    out = {}
    cut = [min] + cut + [max]
    tcut = list(cut)
    for i in range(len(cut)):
        tcut[i] = (tcut[i] - min) / (max - min)
    if isinstance(x, dict):
        pars = x
        if method == "ll":
            N = pars["etl"]
        else:
            N = pars["atl"]
    else:
        if method == "ll":
            Nnotrounded = etl(stats.mean(x), stats.variance(x), reliability, min, max)
            N = round(Nnotrounded)
            pars = betaparameters(x, N, 0, model, l, u)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, N, 0, 2, l, u)
            pars["etl"] = Nnotrounded
            pars["etl rounded"] = N
            pars["lords_k"] = 0
            for i in range(len(cut)):
                cut[i] = tcut[i] * N
                cut[i] = round(cut[i])
        else:
            N = max 
            K = k(stats.mean(x), stats.variance(x), reliability, N)
            pars = betaparameters(x, N, K, model, l, u)
            if (failsafe == True and model == 4) and (l < 0 or u > 1):
                pars = betaparameters(x, max, N, 2, l, u)
            pars["lords_k"] = K
    if "parameters" in output:
        out["parameters"] = pars
        if print_output:
            print("Model parameter estimates:")
            print(out["parameters"])
            print("")
    choose_values = [choose_functions(N, i) for i in range(N + 1)] # TODO: make appropriate changes below.
    if "accuracy" in output:
        print("Estimating accuracy...\n")
        confmat = np.zeros((N + 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(N + 1):
                confmat[j, i] = bbintegrate1(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, j, pars["lords_k"], tcut[i], tcut[i + 1], method)[0]
        confusionmatrix = np.zeros((len(cut) - 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(len(cut) - 1):
                if i != len(cut) - 2:
                    confusionmatrix[i, j] = sum(confmat[cut[i]:cut[i + 1], j])
                else:
                    confusionmatrix[i, j] = sum(confmat[cut[i]:, j])
        accuracy = []
        for i in range(len(cut) - 1):
            accuracy = accuracy + [confusionmatrix[i, i]]
        accuracy = sum(accuracy)
        out["Confusion matrix"] = pd.DataFrame(confusionmatrix)
        out["Overall accuracy"] = accuracy
        if print_output:
            print("Confusion matrix:")
            print(out["Confusion matrix"])
            print("")
            print("Overall accuracy:")
            print(out["Overall accuracy"])
            print("")
    
    if "consistency" in output:
        print("Estimating consistency...\n")
        consmat = np.zeros((N + 1, N + 1))
        for i in range(N + 1):
            for j in range(N + 1):
                if i <= j:
                    consmat[i, j] = bbintegrate2(pars["alpha"], pars["beta"], pars["l"], pars["u"], N, i, j, pars["lords_k"], 0, 1, method)[0]
        lower_triangle = np.tril_indices(consmat.shape[0], 0)
        consmat[lower_triangle] = consmat.T[lower_triangle]
        consistencymatrix = np.zeros((len(cut) - 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(len(cut) - 1):
                if i == 0 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], 0:cut[j + 1]]))
                if i == 0 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1]]))
                if i == 0  and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                if (i != 0 and i != len(cut) - 2) and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], 0:cut[j + 1]]))
                if (i != 0 and i != len(cut) - 2) and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1]]))
                if (i != 0 and i != len(cut) - 2) and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                if i == len(cut) - 2 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, 0:cut[j + 1]]))
                if i == len(cut) - 2 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1]]))
                if i == len(cut) - 2 and j == len(cut) - 2:
                        consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1] + 1]))
            consistency = []
            for i in range(len(cut) - 1):
                consistency = consistency + [consistencymatrix[i, i]]
            consistency = sum(consistency)
        out["Consistency matrix"] = pd.DataFrame(consistencymatrix)
        out["Overall consistency"] = consistency
        if print_output:
            print("Consistency matrix:")
            print(out["Consistency matrix"])
            print("")
            print("Overall consistency:")
            print(out["Overall consistency"])
    return out

# Setting the seed
np.random.seed(123456)

# Define the parameters for the beta distribution
a, b = 6, 4
# The first two parameters are for the location and scale parameters respectively
p_success = rbeta4p(100000, 6, 4, .15, .85)

# Preallocate a matrix of zeros with 1000 rows and 20 columns
rawdata = np.zeros((100000, 100))

# Loop over the columns
for i in range(100000):
    for j in range(100):
    # For each column, generate binomially distributed data
        rawdata[i, j] = np.random.binomial(1, p_success[i], 1)
sumscores = np.sum(rawdata, axis = 1)
meanscores = np.mean(rawdata, axis = 1)

output = cac(sumscores, cronbachs_alpha(rawdata), 0, 100, [50, 75])