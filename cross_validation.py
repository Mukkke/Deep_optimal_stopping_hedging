import numpy as np
from train_val import test_value_info, train_NN
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterSampler
import json
import os


def K_fold_test(X, S, params, k_folds=5):
    kf = KFold(n_splits=k_folds)

    V_hats = []
    all_losses = []

    lr = params['lr']
    threshold = params['threshold']
    model_type = params['model_type']

    for train_index, val_index in kf.split(np.transpose(X, (1, 0, 2))):
        X_train, X_val = X[:, train_index, :], X[:, val_index, :]
        mods, losses_train, times_train = train_NN(S, X_train, model_type=model_type, threshold=threshold, lr=lr)
        V_hat, lower_Y, upper_Y, all_loss_test, all_time_test = test_value_info(S, X_val, mods, model_type=model_type,
                                                                           threshold=threshold)
        all_losses.append(all_loss_test)
        V_hats.append(V_hat)

    mean_loss = np.mean(all_losses, axis=0)
    mean_V_hat = np.mean(V_hats, axis=0)
    return mean_V_hat, mean_loss


def cross_validation(X, S, param_dist, k_folds=5, n_iter=10, log_file="cv_log.json"):
    param_sampler = ParameterSampler(param_dist, n_iter=n_iter)
    best_param = []
    best_V_hat = 0
    log_data = []

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Existing log file is empty or corrupted.")

    for params in param_sampler:
        print(params)
        mean_V_hat, mean_loss = K_fold_test(X, S, params, k_folds=k_folds)

        mean_V_hat = mean_V_hat.tolist()
        mean_loss = mean_loss.tolist()

        log_entry = {
            "params": str(params),
            "mean_V_hat": mean_V_hat,
            "mean_loss": mean_loss
        }
        log_data.append(log_entry)

        if best_V_hat < mean_V_hat:
            best_param = params
            best_V_hat = mean_V_hat

    best_entry = {
        "best_param": best_param,
        "best_V_hat": best_V_hat
    }
    log_data.append(best_entry)

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

    return best_V_hat, best_param
