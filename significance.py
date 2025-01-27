import scipy.stats as stats
import numpy as np


def confidence_interval(data):
    sample_mean = np.mean(data)
    stdev = np.std(data, ddof=1)
    n = len(data)
    degfree = n-1
    confidence = 0.95
    t_score = stats.t.ppf((1+confidence)/2,degfree)
    margin_of_error = t_score * (stdev / np.sqrt(n))

    conf_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    print(conf_interval)
    
def significance(data):
    pop_mean = 0
    t_statistic, p_value = stats.ttest_1samp(data, pop_mean)
    alpha = 0.05

    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    if p_value < alpha:
        print("Result is statistically significant")

    else:
        print("Result is not significant")