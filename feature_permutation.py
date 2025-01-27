import numpy as np
from sklearn.metrics import mean_squared_error

# Assuming the following are defined in your code:
# model - your trained PyTorch model
# x_test - your test set
# y_test_t, y_test_c - the true labels for your test set
# The model's performance function (e.g., RMSE)

# Calculate the baseline performance
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Ensure no gradients are computed
    y_test_pred = model(x_test).cpu().detach().numpy()
baseline_performance = mean_squared_error(y_test_pred, y_test_t)

# Initialize a list to hold the feature importances
feature_importances = []

# Calculate the importance for each feature
for i in range(x_test.shape[1]):  # Iterate over each feature
    # Save the original feature
    original_feature = x_test[:, i].clone()
    # Permute the feature
    permuted_feature = original_feature[torch.randperm(original_feature.size(0))]
    x_test[:, i] = permuted_feature
    
    # Calculate performance with the permuted data
    with torch.no_grad():
        y_test_pred_permuted = model(x_test).cpu().detach().numpy()
    permuted_performance = mean_squared_error(y_test_pred_permuted, y_test_t)
    
    # Calculate the importance as the change in performance
    importance = baseline_performance - permuted_performance
    feature_importances.append(importance)
    
    # Restore the original feature
    x_test[:, i] = original_feature

# Rank the features by their importance
sorted_features = np.argsort(feature_importances)[::-1]  # Indices of features, sorted by importance

# Print the feature importances
print("Feature importances:")
for i, feature_index in enumerate(sorted_features):
    print(f"Feature {feature_index}: Importance {feature_importances[feature_index]}")
