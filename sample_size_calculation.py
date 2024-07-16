import statsmodels.stats.power as smp

# Define parameters
effect_size = 0.5  # Medium effect size
alpha = 0.05 / 107  # Adjusted significance level for multiple comparisons
power = 0.80  # Desired power
ratio = 1  # Ratio of sample size in each group

# Calculate the sample size
sample_size = smp.tt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=ratio)
print(f"Required sample size per group: {sample_size:.2f}")