from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
import pickle

# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

# get standard models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.cluster import KMeans
import numpy as np


from datetime import timedelta

def data_processing(file, undersampled, include_propensity, machine, intervention, outcome, target = 'OutcomeT1', treatment = 'treatment'):
    # change file name to include targets 
    #FILE = f"{FILE}_{treatment}-{target}.csv" 
    data = pd.read_csv(file)

    if undersampled:
        freq_treated = len(data[data['treatment']==1])
        control = data[data['treatment']==0].sample(freq_treated)
        treated = data[data['treatment']==1]
        data = pd.concat((control, treated))

    print(f'DF lenght after undersampling: {len(data)}')

    # isolate covariates
    X = data.drop(columns = [target, 'treatment']).columns
    outcomes = target #f'{target}_t1'
    # apply min-max normalisation
    #data[X] = (data[X] - data[X].min()) / (data[X].max() - data[X].min())

    # drop nan columns
    if len(data.dropna(axis=1)) > 0:
        data = data.dropna(axis=1)
    
    # get train-test, conditioned on treatment
    y_control = data.query('treatment==0')[outcomes]
    X_control = data.drop(columns=outcomes).query('treatment==0')

    y_treated = data.query('treatment==1')[outcomes]
    X_treated = data.drop(columns=outcomes).query('treatment==1')

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_control, y_control, test_size = 0.2)
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_treated, y_treated, test_size = 0.2)
    
    
    # predict the propensity score
    X_train = pd.concat((X_train_c, X_train_t))

    X = X_train.drop(columns = ['treatment']).columns
    ps_model = LogisticRegression(C=1e2, max_iter=10000).fit(X_train[X], X_train['treatment'])
    
    #if include_propensity:

    X_train_c['ps'] = ps_model.predict_proba(X_train_c[X])[:, 1]
    X_train_t['ps'] = ps_model.predict_proba(X_train_t[X])[:, 1]

    X_test_c['ps'] = ps_model.predict_proba(X_test_c[X])[:, 1]
    X_test_t['ps'] = ps_model.predict_proba(X_test_t[X])[:, 1]
    
    if include_propensity == False:
        X_train_c = X_train_c.drop(columns = ['ps'])
        X_train_t = X_train_t.drop(columns = ['ps'])
        
        X_test_c = X_test_c.drop(columns = ['ps'])
        X_test_t = X_test_t.drop(columns = ['ps'])
        
    X_train_c = X_train_c.drop(columns = ['treatment'])
    X_train_t = X_train_t.drop(columns = ['treatment'])
    
    X_test_c = X_test_c.drop(columns = ['treatment'])
    X_test_t = X_test_t.drop(columns = ['treatment'])

    filename_prob_score_model = f"storage/propensity_score_model_{intervention}-{outcome}.sav"

    pickle.dump(ps_model, open(filename_prob_score_model, 'wb'))
    
    return X_train_c, X_test_c, y_train_c, y_test_c, X_train_t, X_test_t, y_train_t, y_test_t

#X_train_c, X_test_c, y_train_c, y_test_c, X_train_t, X_test_t, y_train_t, y_test_t = data_processing(file = FILE)

def doubly_robust(df, X, T, Y):
    ps = LogisticRegression(C=1e2, max_iter=10000).fit(df[X], df[T]).predict_proba(df[X])[:, 1]
    mu0 = LinearRegression().fit(df.query(f"{T}==0")[X], df.query(f"{T}==0")[Y]).predict(df[X])
    mu1 = LinearRegression().fit(df.query(f"{T}==1")[X], df.query(f"{T}==1")[Y]).predict(df[X])
    
    print(f"Estimated outcome under control {np.mean(mu0)}.")
    print(f"Estimated outcome under treatment {np.mean(mu1)}.")
    
    return (
        np.mean(df[T]*(df[Y] - mu1)/ps + mu1) -
        np.mean((1-df[T])*(df[Y] - mu0)/(1-ps) + mu0)
    )

def visualise(predictions_t, predictions_c, y_test_t, y_test_c, machine, target, intervention):
    sns.set_palette('Set2')
    sns.set_theme(style="whitegrid")#sns.color_palette('vlag')
    
    # predictions into dataframe and append
    pred_t = pd.DataFrame([predictions_t, y_test_t, ['treated' for i in range(len(y_test_t))]]).T
    pred_c = pd.DataFrame([predictions_c, y_test_c, ['control' for i in range(len(y_test_c))]]).T
    pred = pred_t.append(pred_c)
    pred.columns = ['Predicted', 'True', 'Type']

    # separate predicted and true
    predicted = pred['Predicted']
    true = pred['True']

    # plot
    plt.figure(figsize=(12,6))
    sns.scatterplot(x=predicted, y=true, hue = pred['Type'])
    plt.title(f'Predicted vs. True Outcomes for {intervention}-{target} using {machine}')
    plt.legend()

    # store result
    #plt.savefig(f"results/{machine}/Meta{intervention}-{target}.png", dpi=300)    
    
    plt.show()
    
def visualise_ites(ites, machine, target, intervention, split="test"): 
    ites = [i[0] for i in ites.reshape(-1,1)]
    ites = sorted(list(ites))

    x_values = range(1, len(ites) + 1)
    stdev = np.std(ites)
    upper_bound = [val + stdev for val in ites]
    lower_bound = [val - stdev for val in ites]

    sns.set(style="whitegrid")

    #plt.figure(figsize=(8,4))
    plt.figure(figsize=(12,6))

    plt.fill_between(x_values, lower_bound, upper_bound, color='gray', alpha=0.2, label="standard deviation")
    plt.axhline(y=np.mean(ites), color='red', linestyle='--', label='ATE estimate')
    plt.xlabel("Ordered Patient Indices")
    plt.ylabel(target)
    plt.title(f"Orded ITE-estimates for {intervention}-{target} pair on {split}.")

    sns.lineplot(x=x_values, y=ites, marker='o', color='b', label='predicted ITE values', ci=stdev)
    plt.savefig(f"plots/{machine}-{intervention}-{target}.png", dpi=300)

    plt.show()
    
def undersample(df, kind = "control"):
    # undersample control
    if kind == "control":
        # number of treated
        freq_treated = len(df[df['treatment']==1])
        control = df[df['treatment']==0].sample(freq_treated)
        treated = df[df['treatment']==1]
    elif kind == "treated":
        freq_control = len(df[df['treatment']==0])
        control = df[df['treatment']==0]
        treated = df[df['treatment']==1].sample(freq_control)

    df = pd.concat((control, treated))
    
    return df



# Function to calculate prediction intervals
def get_prediction_intervals(predictions, percentile=95):
    lower_bound = np.percentile(predictions, (100 - percentile) / 2.0, axis=0)
    upper_bound = np.percentile(predictions, 100 - (100 - percentile) / 2.0, axis=0)
    return lower_bound, upper_bound

# Function to compute bootstrapped confidence intervals
def bootstrap_confidence_intervals(model, X, y, n_bootstrap=100, percentile=95):
    predictions_bootstrap = []
    for i in range(n_bootstrap):
        #print(i)
        bootstrap_indices = np.random.choice(range(len(X)), size=len(X), replace=True)
        X = np.array(X)
        y = np.array(y)
        X_bootstrap = X[bootstrap_indices]
        model.fit(X_bootstrap, y[bootstrap_indices])
        predictions_bootstrap.append(model.predict(X))
    predictions_bootstrap = np.array(predictions_bootstrap)
    return get_prediction_intervals(predictions_bootstrap, percentile)
    
def run_model(file, model0, model1, machine, intervention, outcome, include_propensity, undersampled = False, num_iter = 1):
    # if more iterations are necessary
    maec, maet, msec, mset, rtc, rtt = [], [], [], [], [], []
    
    X_train_c, X_test_c, y_train_c, y_test_c, X_train_t, X_test_t, y_train_t, y_test_t = data_processing(file = file, machine = machine, include_propensity = include_propensity, outcome = outcome, intervention = intervention, undersampled = undersampled)

    
    f_c = model0
    f_c.fit(X_train_c, y_train_c)
    predictions_c = f_c.predict(X_test_c)
    # treated
    f_t = model1
    f_t.fit(X_train_t, y_train_t)

    predictions_t = f_t.predict(X_test_t)

    maec.append(mean_absolute_error(predictions_c, y_test_c))
    maet.append(mean_absolute_error(predictions_t, y_test_t))
    msec.append(mean_squared_error(predictions_c, y_test_c))
    mset.append(mean_squared_error(predictions_t, y_test_t))
    rtc.append(r2_score(predictions_c, y_test_c))
    rtt.append(r2_score(predictions_t, y_test_t))

    # compute ITEs for test
    ites_t = f_t.predict(X_test_t) - f_c.predict(X_test_t)
    ites_c = f_t.predict(X_test_c) - f_c.predict(X_test_c)
    # compute Individual treatment effects
    ites_test = np.append(ites_t, ites_c)

    # compute ITEs for train
    ites_t_train = f_t.predict(X_train_t) - f_c.predict(X_train_t)
    ites_c_train = f_t.predict(X_train_c) - f_c.predict(X_train_c)
    # compute Individual treatment effects
    ites_train = np.append(ites_t_train, ites_c_train)

    # compute relevant errors and record 
    results = {'MAE_C': np.mean(maec),
               'MAE_T': np.mean(maet),
                'MSE_C': np.mean(msec),
                'MSE_T': np.mean(mset),
                 'R2_C': np.mean(rtc),
                 'R2_T': np.mean(rtt)}

    hyperparameters = f_t.get_params()
    print(hyperparameters)

    # Convert to DataFrame
    df = pd.DataFrame([hyperparameters])
    df.to_csv(f'hyperparameters/model_hyperparameters_{machine}.csv')

    metrics = pd.DataFrame.from_dict(results, orient = 'index', columns = [machine])
    metrics.to_csv(f"results/model_results_{machine}_{intervention}-{outcome}.csv")
    
    filename_t = f"storage/model_t_{machine}_{intervention}-{outcome}.sav"
    filename_c = f"storage/model_c_{machine}_{intervention}-{outcome}.sav"

    # save models
    pickle.dump(f_t, open(filename_t, 'wb'))
    pickle.dump(f_c, open(filename_c, 'wb'))

    
    # Calculate confidence intervals for predictions
    lower_c, upper_c = bootstrap_confidence_intervals(f_c, X_test_c, y_test_c)
    lower_t, upper_t = bootstrap_confidence_intervals(f_t, X_test_t, y_test_t)
    
        
    # Print prediction intervals
    print("Control group prediction intervals:")
    print("Lower:", lower_c)
    print("Upper:", upper_c)

    print("\nTreatment group prediction intervals:")
    print("Lower:", lower_t)
    print("Upper:", upper_t)

    # Compute ITEs for test
    ites_t = f_t.predict(X_test_t) - f_c.predict(X_test_t)
    ites_c = f_t.predict(X_test_c) - f_c.predict(X_test_c)

    # Compute individual treatment effects
    ites_test = np.concatenate([ites_t, ites_c])


    return metrics, predictions_t, predictions_c, y_test_t, y_test_c, ites_test, ites_train, X_test_t, X_test_c



def run_model_class(file, model0, model1, machine, intervention, outcome, include_propensity, undersampled = False, num_iter = 1):
    # if more iterations are necessary
    acc_c, acc_t, f1_c, f1_t = [], [], [], []
    
    X_train_c, X_test_c, y_train_c, y_test_c, X_train_t, X_test_t, y_train_t, y_test_t = data_processing(file = file, machine = machine, include_propensity = include_propensity, outcome = outcome, intervention = intervention, undersampled = undersampled)

    
    f_c = model0
    f_c.fit(X_train_c, y_train_c)
    predictions_c = f_c.predict(X_test_c)
    predictions_c_proba = f_c.predict_proba(X_test_c)
    # treated
    f_t = model1
    f_t.fit(X_train_t, y_train_t)
    predictions_t = f_t.predict(X_test_t)
    predictions_t_proba = f_t.predict_proba(X_test_t)

    acc_c.append(accuracy_score(predictions_c, y_test_c))
    acc_t.append(accuracy_score(predictions_t, y_test_t))
#     f1_c.append(f1_score(predictions_c, y_test_c))
#     f1_t.append(f1_score(predictions_t, y_test_t))

    # compute ITEs for test
    ites_t = f_t.predict(X_test_t) - f_c.predict(X_test_t)
    ites_c = f_t.predict(X_test_c) - f_c.predict(X_test_c)
    # compute Individual treatment effects
    ites_test = np.append(ites_t, ites_c)

    # compute ITEs for train
    ites_t_train = f_t.predict(X_train_t) - f_c.predict(X_train_t)
    ites_c_train = f_t.predict(X_train_c) - f_c.predict(X_train_c)
    # compute Individual treatment effects
    ites_train = np.append(ites_t_train, ites_c_train)

    # compute relevant errors and record 
    results = {'ACC_C': np.mean(acc_c),
               'ACC_T': np.mean(acc_t)}

    hyperparameters = f_t.get_params()
    print(hyperparameters)

    # Convert to DataFrame
    df = pd.DataFrame([hyperparameters])
    df.to_csv(f'hyperparameters/model_hyperparameters_{machine}.csv')

    metrics = pd.DataFrame.from_dict(results, orient = 'index', columns = [machine])
    metrics.to_csv(f"results/model_results_{machine}_{intervention}-{outcome}.csv")
    
    filename_t = f"storage/model_t_{machine}_{intervention}-{outcome}.sav"
    filename_c = f"storage/model_c_{machine}_{intervention}-{outcome}.sav"

    # save models
    pickle.dump(f_t, open(filename_t, 'wb'))
    pickle.dump(f_c, open(filename_c, 'wb'))

    
#     # Calculate confidence intervals for predictions
#     lower_c, upper_c = bootstrap_confidence_intervals(f_c, X_test_c, y_test_c)
#     lower_t, upper_t = bootstrap_confidence_intervals(f_t, X_test_t, y_test_t)
    
        
#     # Print prediction intervals
#     print("Control group prediction intervals:")
#     print("Lower:", lower_c)
#     print("Upper:", upper_c)

#     print("\nTreatment group prediction intervals:")
#     print("Lower:", lower_t)
#     print("Upper:", upper_t)

    # Compute ITEs for test
    ites_t = f_t.predict(X_test_t) - f_c.predict(X_test_t)
    ites_c = f_t.predict(X_test_c) - f_c.predict(X_test_c)

    # Compute individual treatment effects
    ites_test = np.concatenate([ites_t, ites_c])
    
    # Compute ITEs for test
    ites_t_proba = f_t.predict_proba(X_test_t) - f_c.predict_proba(X_test_t)
    ites_c_proba = f_t.predict_proba(X_test_c) - f_c.predict_proba(X_test_c)

    # Compute individual treatment effects
    ites_test_proba = np.concatenate([ites_t_proba, ites_c_proba])
    

    return metrics, predictions_t, predictions_c, y_test_t, y_test_c, ites_test, ites_train, X_test_t, X_test_c, ites_test_proba




def cv_model(k, model0, model1, file, param_grid):
    X_train_c, X_test_c, y_train_c, y_test_c, X_train_t, X_test_t, y_train_t, y_test_t = data_processing(file = file)

    # Perform GridSearchCV with cross-validation on model 0
    grid_search0 = GridSearchCV(estimator=model0, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    score_model0 = cross_val_score(model0, X_train_c, y_train_c, cv = k, scoring = 'neg_mean_squared_error')
    grid_search0.fit(X_train_c, y_train_c)  # X and y are your dataset and target values

    # Get the best hyperparameters and model
    best_params0 = grid_search0.best_params_
    best_model0 = grid_search0.best_estimator_

    # Train the best model on the entire dataset (if needed)
    best_model0.fit(X_train_c, y_train_c)
    predictions_c = best_model0.predict(X_test_c)

    # Perform GridSearchCV with cross-validation on model 1
    grid_search1 = GridSearchCV(estimator=model1, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    score_model1 = cross_val_score(model1, X_train_t, y_train_t, cv = k, scoring = 'neg_mean_squared_error')
    grid_search1.fit(X_train_t, y_train_t)  # X and y are your dataset and target values

    # Get the best hyperparameters and model
    best_params1 = grid_search1.best_params_
    best_model1 = grid_search1.best_estimator_

    # Train the best model on the entire dataset (if needed)
    best_model1.fit(X_train_t, y_train_t)
    predictions_t = best_model1.predict(X_test_t)

    # compute ITEs for test
    ites_t = best_model1.predict(X_test_t) - best_model0.predict(X_test_t)
    ites_c = best_model1.predict(X_test_c) - best_model0.predict(X_test_c)
    # compute Individual treatment effects
    ites_test = np.append(ites_t, ites_c)
    
    # compute ITEs for test
    ites_t_train = best_model1.predict(X_train_t) - best_model0.predict(X_train_t)
    ites_c_train = best_model1.predict(X_train_c) - best_model0.predict(X_train_c)
    # compute Individual treatment effects
    ites_train = np.append(ites_t_train, ites_c_train)
    
    return np.mean(-score_model0), np.mean(-score_model1), ites_test, ites_train, y_test_t, y_test_c, predictions_t, predictions_c

def impute_missing_values_knn(df, n_neighbors=5):
    # Copy the original DataFrame to avoid modifying it
    imputed_df = df.copy()

    # Initialize KNNImputer with the specified number of neighbors
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Perform KNN imputation on the DataFrame
    imputed_array = imputer.fit_transform(imputed_df.values)

    # Convert the imputed array back to a DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=imputed_df.columns)

    return imputed_df

def value_based_contra_indications_tracker(df, value_based_contra_indications):
    # initialize empty tracker
    contra_indication = lambda x: 1 if x > 0 else 0
    index_tracker = []

    # iterate over columns and corresponding values 
    for key, value in value_based_contra_indications.items():    
        patients_to_be_dropped = list(df[df[key]>value].index)
        for i in patients_to_be_dropped:
            index_tracker.append(i)

    df = df.drop(index=list(set(index_tracker)))
    print(f'{len(set(index_tracker))} observations with value-based contra-indications')
    
    return df

def full_contra_indications_tracker(df, contra_indications):
    # initialize empty tracker
    contra_indication = lambda x: 1 if x > 0 else 0
    index_tracker = []
    # check if there are selected counterindications
    for col in contra_indications:
        # first check if contra indication applies
        contra_ind = df[col].apply(contra_indication)
        # mark items with a contra-indication
        for i in list(contra_ind[contra_ind==1].index):
            index_tracker.append(i)
    
    print(f'{len(set(index_tracker))} observations with contra-indications')
    df = df.drop(index=list(set(index_tracker)))
    
    return df

def period_decomposition(df, target):
    # create temp df
    temp_df = pd.DataFrame(columns = list(df.columns) + ['OutcomeT0', 'OutcomeT1', 'OutcomeT0Date', 
                                                         'OutcomeT1Date'])#, 'AssesmentNumber'])
    counter = 0
    total = len(set(list(df['Clientid'])))

    # for each patient
    for i in set(list(df['Clientid'])):
        counter += 1
        if (counter % 1000) == 0:
            print(f'{counter} out of {total} items completed...')
        date_list = list(df[df['Clientid']==i]['iA9'])

        # for each date
        for j in range(len(date_list)):
            date_baseline = date_list[j]

            # check if end of list is reached
            if j+1 < len(date_list):
                date_followup = date_list[j+1]
                subdf = df[df['Clientid']==i][df[df['Clientid']==i]['iA9']==date_baseline]
                subdf['OutcomeT0'] = float(df[df['Clientid']==i][df[df['Clientid']==i]['iA9']==date_baseline][target])
                subdf['OutcomeT1'] = float(df[df['Clientid']==i][df[df['Clientid']==i]['iA9']==date_followup][target])
                subdf['OutcomeT0Date'] = date_baseline
                subdf['OutcomeT1Date'] = date_followup
                #subdf['AssesmentNumber'] = float(j)

                # append to temp df
                temp_df = temp_df.append(subdf)
            else:
                pass
            
    print("Completed.")
    
    return temp_df

def multicol(df, correlation_threshold):
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Find pairs of features with correlations greater than +0.25 or less than -0.25
    high_correlation_pairs = []

    # create pair list
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if (abs(correlation_matrix.iloc[i, j]) > correlation_threshold) and (correlation_matrix.iloc[i, j] != 1):
                high_correlation_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    # Remove one feature from each high-correlation pair
    for pair in high_correlation_pairs:
        if pair[0] in df.columns and pair[1] in df.columns:
            if (pair[0] != 'OutcomeT1') and (pair[0] != 'treatment') and (pair[0] != 'OutcomeT0'):
                df.drop(columns=[pair[0]], inplace=True)
            elif (pair[1] != 'OutcomeT1') and (pair[1] != 'treatment') and (pair[1] != 'OutcomeT0'):
                df.drop(columns=[pair[1]], inplace=True)
            #df.drop(columns=[pair[0]], inplace=True)
            
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
            
    return df