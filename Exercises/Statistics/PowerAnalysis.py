
import statsmodels.stats.power as smp
import numpy as np
import pandas as pd
import scipy.stats

effectSize = 0.5  # ratio mean/distance of 0.5 between the distributions

""" Perform power analysis to get sample size
    it in order to reject the Null Hypothesis for fair sample selection """

power_analysis = smp.TTestIndPower()
sampleSize = power_analysis.solve_power(
    effect_size=effectSize, power=0.9, alpha=0.05)
# round up from estimated sample size
sampleSize = np.int(np.ceil(sampleSize))
print('sampleSize = ', sampleSize)


""" Given two data groups 
    Monte Carlo simulation of power analysis.
    Πρεπει να τρεξει πάρα πολλες φορές, ωστε απο κεντρικο οριακο θεώρημα
    να γίνει δειγματοληψία απο κανονική κατανομή
    για αντικειμενικό μεσο. Τότε θα ξέρουμε το πραγματικό
    πληθος samples που πρεπει να πάρουμε. (monte carlo)
    
    given:  sampleSize, effectSize
    
    output: power for rejecting the Null Hypothesis
"""



# Create a function that will generate samples and test for
# a difference between groups using a two-sample t-test

def get_t_result(sampleSize, effectSize):
    """
    perform a ttest on random data of n=sampSize
    """

    group1 = np.random.normal(loc=0.0, scale=1.0, size=sampleSize)
    group2 = np.random.normal(loc=effectSize, scale=1.0, size=sampleSize)
    ttresult = scipy.stats.ttest_ind(group1, group2)
    return (ttresult.pvalue)


# simulate runs for sampleSize, more runs means more data.
num_runs = 10000
power_sim_results = pd.DataFrame({'p_value': np.zeros(num_runs)})

for run in range(num_runs):
    power_sim_results.loc[run, 'p_value'] = get_t_result(sampleSize, effectSize)

p_reject = np.mean(power_sim_results['p_value'] < 0.05)
print("p_reject")
print(p_reject)