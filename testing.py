
## TESTS: ###
# Setting the seed
np.random.seed(1234)
from support_functions.betafunctions import cronbachs_alpha, rbeta4p
N_resp, N_items, alpha, beta, l, u = 250, 100, 6, 4, .15, .85
p_success = rbeta4p(N_resp, alpha, beta, l, u)
rawdata = [np.random.binomial(1, p_success[i], N_items) for i in range(N_resp)]
sumscores = [int(i) for i in list(np.sum(rawdata, axis = 1))]
bb = bbclassify(data = sumscores, reliability = cronbachs_alpha(rawdata), min_score = 0, max_score = 100, cut_scores = [50], method = "hb")
print(bb.Parameters)
bb = bbclassify(data = sumscores, reliability = cronbachs_alpha(rawdata), min_score= 0, max_score = 100, cut_scores = [50], model = 2, l = 0.25, u = 0.85, method = "ll")
print(bb.Parameters)
"""
print(bb_hb.Parameters)
bb_hb.modelfit()
print([bb_hb.Modelfit_chi_squared, bb_hb.Modelfit_degrees_of_freedom, bb_hb.Modelfit_p_value])
bb_hb.accuracy()
print(bb_hb.Accuracy)
bb_hb.consistency()
print(bb_hb.Consistency)
"""

"""
bb = bbclassify(data = sumscores, reliability = cronbachs_alpha(rawdata), min_score= 0, max_score = 100, cut_scores = [50, 75], method = "ll")
print(bb._betaparameters(sumscores, bb._calculate_etl(stats.mean(sumscores), stats.variance(sumscores), cronbachs_alpha(rawdata), 0, 100), 0, 4))
bb = bbclassify(data = sumscores, reliability = cronbachs_alpha(rawdata), min_score= 0, max_score = 100, cut_scores = [50, 75], model = 2, l = 0.25, u = 0.85, method = "ll")
print(bb.Parameters)
#p_success = rbeta4p(10000, 5, 3, .25, .75)
#print(stats.mean(p_success))
#print(stats.mean([int(i) for i in np.random.binomial(1, p_success, 10000)]))
"""