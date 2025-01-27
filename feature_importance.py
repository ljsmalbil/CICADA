from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

importance = np.abs(lasso.coef_)

feature_names = np.array(feature_column_names)  # replace with your column names
important_features = feature_names[importance > 0]
sorted_idx = np.argsort(importance[importance > 0])[::-1]
sorted_important_features = important_features[sorted_idx]

print("Most important features (in order):")
for feature in sorted_important_features:
    print(feature)
