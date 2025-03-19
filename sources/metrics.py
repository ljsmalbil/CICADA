import torch
import numpy as np
import pandas as pd

to_numpy = lambda x: x.cpu().detach().numpy()

def pehe_eval(x_test_c, x_test_t, y_0_test, y_1_test, model):
    # Predict regular test values
    y_0_hat_c = model.forward_control(x_test_c)
    y_0_hat_t = model.forward_control(x_test_t)
    y_0_pred = np.concatenate((to_numpy(y_0_hat_t), to_numpy(y_0_hat_c)))
    
    # Predict regular test values
    y_1_hat_c = model.forward_treated(x_test_c)
    y_1_hat_t = model.forward_treated(x_test_t)
    y_1_pred = np.concatenate((to_numpy(y_1_hat_t), to_numpy(y_1_hat_c)))
    
    # Compute Predicted ITE and true ITE
    ite_true = y_1_test - y_0_test
    ite_pred = y_1_pred - y_0_pred    
    pehe = np.sqrt(np.mean(np.square(ite_true - ite_pred)))
    
    return pehe, ite_pred