# import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# get standard models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor

from sources.metrics import pehe_eval
from sources.empirical_data import ihdp_loader
from sources.doubly_robust import doubly_robust

from sources.models.TARNet import TARnetICFR

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans
import numpy as np

from auxiliary import data_processing, doubly_robust, visualise, visualise_ites, impute_missing_values_knn, run_model, undersample, full_contra_indications_tracker, value_based_contra_indications_tracker, period_decomposition, multicol

# from datetime import timedelta
import warnings 
warnings.filterwarnings('ignore')

###########################
#                         #
#      PRELIMINARIES      #
#                         #
###########################

# set seed
np.random.seed(42)

# set the exposure threshold for binary dichtomization. In the case of PT, 60 minutes of PT at least. 
EXPOSURE_THRESHOLD = 60

# set the period between observations. We only consider the effects between 60 and 120 days after exposure. 
PERIOD_MIN = 60
PERIOD_MAX = 120

# if we want to move multicollinear columns, set to True
REMOVE_MULTI_COL = True

# set the threshold for multicollinearity drops 
CORRELATION_THRESHOLD = 0.25

# set to True if we want to undersample
UNDERSAMPLE = True

# set to True if we want to impute missing values
IMPUTE = True


###########################
#                         #
#          Data           #
#                         #
###########################

file = "data/dutch_LTCF_all.csv"
treatment = 'in3eb' # (minutes or days of physical therapy)
target = 'sADLSF' 

# read data
df = pd.read_csv(file)
print(f"Treatment before processing {len(df[df[treatment]>EXPOSURE_THRESHOLD])}")


###########################
#                         #
#  Variable Selection     #
#                         #
###########################


# list clinical contra-indications
contra_indications = ['ij2e', 'ij2l', 'ij2n', 'ij2t', 'ij2q', 'ij2r', 'ij6b', 'ij6c',  'in6b', 'in6c'] 

value_based_indications = {'sCPS': 4, 'in2a':2, 'in2a':3, 'in2c': 2, 'in2c': 3, 'in2e': 2, 'in2e': 3, 
                          'in2f': 2, 'in2f': 3, 'in2g': 2, 'in2g': 3, 'in2h': 2, 'in2h': 3, 'in2i': 2, 'in2i':3,
                          'in2j':2, 'in2j':3}

# also convert values to list for ease of processing later on
listed_val_based_ind = [key for key, value in value_based_indications.items()]

contra_indications = contra_indications #+ listed_val_based_ind

# list relevant confounders
confounders = ['ik3','ic3a', 'ic3b', 'id3a', 'id4a', 'ie3e', 'ih1', 'ih2', 
                        'ih3', 'ih5', 'ij6a', 'ik2a',  
                        'il7', 'il1','sCPS']

# list covars that may be relevant but have not been listed elsewhere
other_relevant_covars = ['iA2', 'sAGE_cat', 'ia12a', 'ia13', 'ib5a', 'ib5b', 'ib5c', 'ic1', 'ic2c']

# list clinical indications
clinical_indications = ['sDRS', 'ij1', 'ij12', 'ij2a', 'ij2b', 'ij2c', 'ij2d']

# list of other relevant variables
others = ['iA9', 'Clientid', treatment, target]


###########################
#                         #
#  Indication Handling    #
#                         #
###########################

# select and drop patients with full indications
df = full_contra_indications_tracker(df, contra_indications)
# select and drop based on indication value
df = value_based_contra_indications_tracker(df, value_based_indications)
# select relant covariates
df = df[clinical_indications+confounders + other_relevant_covars + others + contra_indications + listed_val_based_ind]
print(f"Treatment after selecting covariates {len(df[df[treatment]>EXPOSURE_THRESHOLD])}")


###########################
#                         #
#Assesment Period Handling#
#                         #
###########################

# get number of assesments
counter = lambda x: len(df[df['Clientid']==x])

# count number of items
df['num_assesments'] = df['Clientid'].apply(counter)

# get number of assesments higher than 1
df = df[df['num_assesments']>=2]

print(f'{len(df)} observations remaining.')

print(f"Treatment after selecting on number of assesments {len(df[df[treatment]>EXPOSURE_THRESHOLD])}")


# convert column to datetime 
df['iA9'] = pd.to_datetime(df['iA9']) 
# sort values by ID and date
df = df.sort_values(by = ['Clientid', 'iA9'])
# drop nans on dates of assesment
df = df[df['iA9'].isna()==False]
# drop duplicated values
df = df.drop_duplicates()
df[target] = df[target].astype(float)

# drop duplicated assesments
for i in list(df['Clientid']):
    if True in list(df[df['Clientid']==i]['iA9'].duplicated()==True):
        index_true = np.where((df[df['Clientid']==i]['iA9'].duplicated()==True))[0][0]
        remove = list(df[df['Clientid']==i]['iA9'].index)[index_true]
        df = df.drop(index = [remove])
    else:
        continue
        
df = period_decomposition(df, target = target)
print(f"Treatment after processing {len(df[df[treatment]>EXPOSURE_THRESHOLD])}")

###########################
#                         #
#  Treatment Handling     #
#                         #
###########################

# assign temp to df
threshold = EXPOSURE_THRESHOLD   

binary = lambda x: 1 if x >= threshold else 0
# convert treatment to binary
df['treatment'] = df[treatment].apply(binary)  
df = df.drop(columns = [treatment])

# drop rows with missing outcome or treatment 
df = df.dropna(subset = ['OutcomeT0', 'OutcomeT1', 'treatment'])
df.head(3)

###########################
#                         #
#     Period Selection    #
#                         #
###########################

df['date_diff'] = df['OutcomeT1Date'] - df['OutcomeT0Date'] 

df['date_diff'] = df['date_diff'].dt.days 
df = df[df['date_diff'] <= PERIOD_MAX]
df = df[df['date_diff'] >= PERIOD_MIN]

df = df.drop(columns = ['OutcomeT1Date', 'OutcomeT0Date', 'date_diff', 'iA9', target, 'Clientid'])
# examine how many treatment observations are left
len(df[df['treatment']==1])

###########################
#                         #
#   Missing Imputation    #
#                         #
###########################

imputing = IMPUTE
if imputing == True:
    df = impute_missing_values_knn(df, n_neighbors=5)
# else:
#     df = df.dropna()
#     print(len(df[df['treatment']==1]
   


###########################
#                         #
#   Multi.Col Handling    #
#                         #
###########################

    
if REMOVE_MULTI_COL == True:
    df = multicol(df, correlation_threshold)
 
# #                         #
# #      Data Storing       #
# #                         #
# ###########################
          
# treatment = 'in3eb' # (minutes or days of physical therapy)
# target = 'sADLSF' 

# #df = df.drop(columns = 'num_assesments')
# df.to_csv(f"data/03-10-2023-Dutch_LTCF_cleaned_data_with_selected_covar_{treatment}-{target}.csv", index = False)
          
# ###########################
# #                         #
#     #      CONTINUE HERE WITH ML PART       #
# #                         #
# ###########################
          