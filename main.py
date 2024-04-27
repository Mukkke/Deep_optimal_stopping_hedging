import numpy as np
import torch
from train_val import test_value_info, train_NN
from stock_generation import Stock
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint
from cross_validation import cross_validation

import os
import random
import json

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

with open('config.json', 'r') as f:
    config = json.load(f)

T_1 = config["T"]
K_1 = config["K"]
sigma_1 = config["sigma"]
delta_1 = config["delta"]
S0_1 = config["S0"]
r_1 = config["r"]
N_1 = config["N"]
M_1 = config["M"]
K_path_1 = config["K_path"]
d_1 = config["d"]
e_1 = config["e"]

S = Stock(T_1, K_1, sigma_1, delta_1, S0_1, r_1, N_1, M_1, d_1, e_1, K_path_1)
X = torch.from_numpy(S.GBM()).float()

param_dist = {
    'threshold': np.random.uniform(0, 1, size=100),
    'lr': np.logspace(np.log10(0.001), np.log10(0.1), num=100),
    # 'model_type': ['MLP', 'CNN', 'GNN', 'Transformer']
    'model_type': ['CNN']
}

best_V_hat, best_param = cross_validation(X, S, param_dist, k_folds=5, n_iter=10)


print(best_param)

# MLP_mods, losses_MLP_train, times_MLP_train = train_NN(S, "MLP")
# V_hat, lower_Y, upper_Y, all_loss_test, all_time_test = value_info(S, NN_mods)

# LSTM_mods = train_NN(S, "LSTM")
# V_hat, lower_Y, upper_Y = value_info(S, LSTM_mods)

# Transformer_mods, losses_Transformer_train, times_Transformer_train = train_NN(S, "Transformer")
# V_hat, lower_Y, upper_Y, all_loss_test, all_time_test = value_info(S, Transformer_mods)

# GNN_mods, losses_GNN_train, times_GNN_train = train_NN(S, "GNN")
# V_hat, lower_Y, upper_Y, all_loss_test, all_time_test = value_info(S, GNN_mods, "GNN")

# CNN_mods, losses_CNN_train, times_CNN_train = train_NN(S, "CNN", threshold=0.5)
# V_hat, lower_Y, upper_Y, all_loss_test, all_time_test = value_info(S, CNN_mods, "CNN", threshold=0.5)
