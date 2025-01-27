import numpy as np
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor

# Assuming X_train, X_test, y_train, y_test are your data

# Number of bootstrap samples
n_bootstraps = 1000

# Placeholder for predictions
predictions = np.zeros((n_bootstraps, len(X_test)))

# Bootstrapping
for i in range(n_bootstraps):
    # Resample the training data
    X_resampled, y_resampled = resample(X_train, y_train)
    
    # Train the model
    model = RandomForestRegressor()
    model.fit(X_resampled, y_resampled)
    
    # Predict on the test set
    predictions[i, :] = model.predict(X_test)

# Calculate the 95% confidence intervals (2.5th and 97.5th percentiles)
lower_bound = np.percentile(predictions, 2.5, axis=0)
upper_bound = np.percentile(predictions, 97.5, axis=0)

# Display the confidence intervals
for i in range(len(X_test)):
    print(f"Prediction: {np.mean(predictions[:, i]):.2f}, 95% CI: [{lower_bound[i]:.2f}, {upper_bound[i]:.2f}]")
