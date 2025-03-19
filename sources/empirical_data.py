import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def ihdp_loader(file, device, version = 0, test_size = 0.2):
    
    # Define to_tensor auxiliary function
    to_tensor = lambda x: torch.tensor(np.array(x), dtype = torch.float32).to(device)

    # Get Data
    with open(file, 'rb') as trf:
        train_data = np.load(trf)
        y = train_data['yf'][:, version].astype('float32') 
        t = train_data['t'][:, version].astype('float32')
        x = train_data['x'][:,:,version].astype('float32')
        mu_0 = train_data['mu0'][:, version].astype('float32')
        mu_1 = train_data['mu1'][:, version].astype('float32')
        data = {'x': x, 't': t, 'y': y, 't': t, 'mu_0': mu_0, 'mu_1': mu_1}

    # Convert NumPy arrays to Pandas dataframes
    df = pd.DataFrame(data['x'])
    df['y'] = data['y']
    df['t'] = data['t']
    df['mu_0'] = data['mu_0']
    df['mu_1'] = data['mu_1']

    # Isolate groups on covariates
    X_train_t = df[df['t']==1]
    X_train_c = df[df['t']==0]

    # Isolate groups on target
    y_train_t = df[df['t']==1]['y']#, 'mu_0', 'mu_1']
    y_train_c = df[df['t']==0]['y']#, 'mu_0', 'mu_1']

    # Split groups into train and test
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_train_t.drop(columns = ['t', 'y', 'mu_0', 'mu_1']), X_train_t[['y', 'mu_0', 'mu_1']], test_size=test_size, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_train_c.drop(columns = ['t', 'y', 'mu_0', 'mu_1']), X_train_c[['y', 'mu_0', 'mu_1']], test_size=test_size, random_state=42)

    # Process counterfactuals and outcomes
    y_test_c_f, y_test_c_cf, y_test_t_f, y_test_t_cf = y_test_c['mu_0'], y_test_c['mu_1'], y_test_t['mu_1'], y_test_t['mu_0'] 
    y_train_c_f, y_train_c_cf, y_train_t_f, y_train_t_cf = y_train_c['mu_0'], y_train_c['mu_1'], y_train_t['mu_1'], y_train_t['mu_0'] 
    y_train_t, y_train_c = y_train_t['y'], y_train_c['y']

    # Concatenate factual and counterfactuals
    y_0_test = np.concatenate((y_test_t_cf, y_test_c_f))
    y_1_test = np.concatenate((y_test_t_f, y_test_c_cf))
    y_0_train = np.concatenate((y_train_t_cf, y_train_c_f))
    y_1_train = np.concatenate((y_train_t_f, y_train_c_cf))

    # convert output to tensors
    x_t, x_c, y_t, y_c, x_test_t, x_test_c = to_tensor(X_train_t), to_tensor(X_train_c), to_tensor(y_train_t), to_tensor(y_train_c), to_tensor(X_test_t), to_tensor(X_test_c)
    
    return df, x_t, x_c, y_t, y_c, x_test_t, x_test_c, y_0_test, y_1_test, y_0_train, y_1_train