import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to calculate prediction intervals
def get_prediction_intervals(predictions, percentile=95):
    lower_bound = np.percentile(predictions, (100 - percentile) / 2.0, axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - percentile) / 2.0, axis=0)
    return lower_bound, upper_bound

# Function to compute bootstrapped confidence intervals
def bootstrap_confidence_intervals(model, X, n_bootstrap=1000, percentile=95):
    predictions_bootstrap = []
    for _ in range(n_bootstrap):
        bootstrap_indices = np.random.choice(range(len(X)), size=len(X), replace=True)
        X_bootstrap = X[bootstrap_indices]
        model.fit(X_bootstrap, y[bootstrap_indices])
        predictions_bootstrap.append(model.predict(X))
    predictions_bootstrap = np.array(predictions_bootstrap)
    return get_prediction_intervals(predictions_bootstrap, percentile)

# Initialize models
f_c = model0
f_t = model1

# Fit control model
f_c.fit(X_train_c, y_train_c)
predictions_c = f_c.predict(X_test_c)

# Fit treated model
f_t.fit(X_train_t, y_train_t)
predictions_t = f_t.predict(X_test_t)

# Compute errors
maec = mean_absolute_error(predictions_c, y_test_c)
maet = mean_absolute_error(predictions_t, y_test_t)
msec = mean_squared_error(predictions_c, y_test_c)
mset = mean_squared_error(predictions_t, y_test_t)
rtc = r2_score(predictions_c, y_test_c)
rtt = r2_score(predictions_t, y_test_t)

# Calculate confidence intervals for predictions
lower_c, upper_c = bootstrap_confidence_intervals(f_c, X_test_c)
lower_t, upper_t = bootstrap_confidence_intervals(f_t, X_test_t)

# Compute ITEs for test
ites_t = f_t.predict(X_test_t) - f_c.predict(X_test_c)
ites_c = f_t.predict(X_test_c) - f_c.predict(X_test_c)

# Compute individual treatment effects
ites_test = np.concatenate([ites_t, ites_c])

# Compute ITEs for train
ites_t_train = f_t.predict(X_train_t) - f_c.predict(X_train_t)
ites_c_train = f_t.predict(X_train_c) - f_c.predict(X_train_c)
ites_train = np.concatenate([ites_t_train, ites_c_train])

# Compute relevant errors and record
results = {
    'MAE_C': maec,
    'MAE_T': maet,
    'MSE_C': msec,
    'MSE_T': mset,
    'R2_C': rtc,
    'R2_T': rtt
}

# Print results and hyperparameters
print(results)
hyperparameters = f_t.get_params()
print(hyperparameters)

# Print prediction intervals
print("Control group prediction intervals:")
print("Lower:", lower_c)
print("Upper:", upper_c)

print("\nTreatment group prediction intervals:")
print("Lower:", lower_t)
print("Upper:", upper_t)
