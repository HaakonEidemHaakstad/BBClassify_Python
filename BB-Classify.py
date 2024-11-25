from support_functions.betafunctions import *

def cac(x, reliability: float, min: float, max: float, cut: float, model: int = 4, l: float = 0, u: float = 1, failsafe: bool = False, method: str = "ll", output: list[str] = ["parameters", "accuracy", "consistency"], print_output = True):
    """
    Estimate accuracy and consistency of classifications using beta-binomial models.

    This function supports two approaches:
    - Livingston and Lewis (LL)
    - Hanson and Brennan (HB)

    Parameters
    ----------
    x : list of float or int
        Either a list of values to which a beta-binomial model is to be fitted,
        or a list of model parameters.
    reliability : float
        The test-score reliability coefficient.
    min : float
        Minimum possible test score (used only in the Livingston and Lewis approach).
    max : float
        - For the Livingston and Lewis approach: Maximum possible test score.
        - For the Hanson and Brennan approach: Actual test length (number of items).
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
    method : str, optional
        Estimation approach to use:
        - "ll": Livingston and Lewis (default)
        - Any other string: Hanson and Brennan
    output : list of str, optional
        Specifies the analyses to include in the output:
        - "parameters": Model parameter estimates
        - "accuracy": Confusion matrices and accuracy statistics
        - "consistency": Consistency matrices and statistics
        Default is ["parameters", "accuracy", "consistency"].
    print_output : bool, optional
        Whether to print intermediate results to the console. Default is True.

    Returns
    -------
    dict
        A dictionary containing the specified output. Keys may include:
        - "parameters": Model parameter estimates
        - "Confusion matrix": Confusion matrix data
        - "Overall accuracy": Overall classification accuracy
        - "Consistency matrix": Consistency matrix data
        - "Overall consistency": Overall classification consistency

    Raises
    ------
    ValueError
        If inputs are invalid, such as `l` >= `u` in a two-parameter model or other
        parameter conflicts.
"""

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

    choose_values = [choose_functions(N, i) for i in range(N + 1)]

    if "accuracy" in output:
        print("Estimating accuracy...\n")
        confmat = np.zeros((N + 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(N + 1):
                confmat[j, i] = bbintegrate1_2(pars["alpha"], pars["beta"], pars["l"], pars["u"], choose_values[j], N, j, pars["lords_k"], tcut[i], tcut[i + 1], method)[0]
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
                    consmat[i, j] = bbintegrate2_1(pars["alpha"], pars["beta"], pars["l"], pars["u"], choose_values[i], choose_values[j], N, i, j, pars["lords_k"], 0, 1, method)[0]
        lower_triangle = np.tril_indices(consmat.shape[0], 0)
        consmat[lower_triangle] = consmat.T[lower_triangle]
        consistencymatrix = np.zeros((len(cut) - 1, len(cut) - 1))
        for i in range(len(cut) - 1):
            for j in range(len(cut) - 1):
                if i == 0 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], 0:cut[j + 1]]))
                elif i == 0 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1]]))
                elif i == 0  and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[0:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                elif (i != 0 and i != len(cut) - 2) and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], 0:cut[j + 1]]))
                elif (i != 0 and i != len(cut) - 2) and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1]]))
                elif (i != 0 and i != len(cut) - 2) and j == len(cut) - 2:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1], cut[j]:cut[j + 1] + 1]))
                elif i == len(cut) - 2 and j == 0:
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, 0:cut[j + 1]]))
                elif i == len(cut) - 2 and (j != 0 and j != len(cut) - 2):
                    consistencymatrix[i, j] = sum(sum(consmat[cut[i]:cut[i + 1] + 1, cut[j]:cut[j + 1]]))
                else:
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


## TESTS: ###

# Setting the seed
#np.random.seed(123456)

# Define the parameters for the beta distribution
#a, b = 6, 4
# The first two parameters are for the location and scale parameters respectively
#p_success = rbeta4p(100000, 6, 4, .15, .85)

# Preallocate a matrix of zeros with 1000 rows and 20 columns
#rawdata = np.zeros((100000, 100))

# Loop over the columns
#for i in range(100000):
#    for j in range(100):
    # For each column, generate binomially distributed data
#        rawdata[i, j] = np.random.binomial(1, p_success[i], 1)
#sumscores = np.sum(rawdata, axis = 1)
#meanscores = np.mean(rawdata, axis = 1)
#output = cac2(sumscores, cronbachs_alpha(rawdata), 0, 100, [50, 75], output = ["accuracy", "consistency"], print_output = False)
#print(output)
#exit()
#print("starting")
#from time import time
#t1 = time()
#for i in range(10):
#    output = cac(sumscores, cronbachs_alpha(rawdata), 0, 100, [50, 75], output = ["accuracy", "consistency"], print_output = False, method = "")
#    print(f"cac1 time: {time() - t1}, accuracy: {output["Overall accuracy"]}, consistency: {output["Overall consistency"]}")
#    t1 = time()
#    output = cac2(sumscores, cronbachs_alpha(rawdata), 0, 100, [50, 75], output = ["accuracy", "consistency"], print_output = False, method = "")
#    print(f"cac2 time: {time() - t1}, accuracy: {output["Overall accuracy"]}, consistency: {output["Overall consistency"]}")