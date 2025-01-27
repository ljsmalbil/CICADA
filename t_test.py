import numpy as np
from scipy import stats

# Simulated data for treatment and control groups
treatment_group = np.array([25, 30, 35, 40, 45])
control_group = np.array([20, 22, 24, 26, 28])

# Calculate the ATE (Average Treatment Effect)
ate = np.mean(treatment_group) - np.mean(control_group)

# Assuming you have calculated the standard error of the ATE (SE_ATE)
se_ate = 2.0  # Replace with the actual standard error

# Set your significance level (alpha)
alpha = 0.05

# Calculate the test statistic
test_statistic = ate / se_ate

# Calculate the p-value for a two-tailed test
p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))

# Print the results
print("ATE:", ate)
print("Standard Error (SE_ATE):", se_ate)
print("Test Statistic:", test_statistic)
print("P-Value:", p_value)

# Make a decision based on the p-value
if p_value < alpha:
    print("Reject the null hypothesis. There is a statistically significant causal effect.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence of a significant causal effect.")
