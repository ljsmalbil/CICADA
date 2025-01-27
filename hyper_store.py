import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Example models
models = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=2),
    'SVC': SVC(C=1.0, kernel='linear'),
    'LogisticRegression': LogisticRegression(C=1.0, penalty='l2')
}

# File path for the Excel file
excel_file = '/mnt/data/model_hyperparameters.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    for model_name, model in models.items():
        # Get hyperparameters
        hyperparameters = model.get_params()

        # Convert to DataFrame
        df = pd.DataFrame([hyperparameters])

        # Write each model's hyperparameters to a different sheet
        df.to_excel(writer, sheet_name=model_name)

print("Hyperparameters of models have been saved to 'model_hyperparameters.xlsx'.")

