import numpy as np
import torch
from torch_geometric.data import Data
import time
import scipy.stats
from nn_models import mlp_stopping, CNN1D, GNNModel, TransformerModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss(y_pre, s, x, n, tau):
    N, num_path, _ = x.shape
    r_n = torch.zeros(num_path, device=device)
    for m in range(0, num_path):
        r_n[m] = -s.g(n, m, x) * y_pre[m] - s.g(tau[m], m, x) * (1 - y_pre[m])
    return r_n.mean()


def train_NN_stopping_t(n, X, s, tau_n_after, model_type, lr=0.0001, epochs=100, train_split=0.8, early_stopping=True,
                        patience=5):
    losses = []
    times = []
    _, num_path, _ = X.shape
    train_size = int(num_path * train_split)
    X_train = X[:, :train_size, :].to(device)
    X_val = X[:, train_size:num_path, :].to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if model_type == "MLP":
        model = mlp_stopping(s.d, 64, 64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            start_time = time.time()
            # Training
            model.train()
            F = model.forward(X_train[n])
            optimizer.zero_grad()
            criterion = loss(F, s, X_train, n, tau_n_after)
            criterion.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model.forward(X_val[n])
                val_loss = loss(val_output, s, X_val, n, tau_n_after).item()

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print(
                            f'Early stopping at epoch {epoch} as validation loss did not improve for {patience} consecutive epochs.')
                        break

            end_time = time.time()
            epoch_time = end_time - start_time
            times.append(epoch_time)
            losses.append(criterion.item())
        return F, model, losses, times

    elif model_type == "Transformer":
        model = TransformerModel(s.d, 64, 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            start_time = time.time()
            # Training
            model.train()
            F = model.forward(X_train[n])  # Add an extra dimension for batch
            optimizer.zero_grad()
            criterion = loss(F, s, X_train, n, tau_n_after)
            criterion.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model.forward(X_val[n])
                val_loss = loss(val_output, s, X_val, n, tau_n_after).item()

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print(
                            f'Early stopping at epoch {epoch} as validation loss did not improve for {patience} consecutive epochs.')
                        break

            end_time = time.time()
            epoch_time = end_time - start_time
            times.append(epoch_time)
            losses.append(criterion.item())
        return F, model, losses, times

    elif model_type == "GNN":
        edge_index = torch.tensor([
            [i, i + 1] for i in range(s.N - 1)
        ], dtype=torch.long).t().contiguous()
        model = GNNModel(s.d, hidden_dim=64, output_dim=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            start_time = time.time()
            # Training
            model.train()
            data_Xn = Data(x=X_train[n], edge_index=edge_index)
            F = model(data_Xn)
            optimizer.zero_grad()
            criterion = loss(F, s, X_train, n, tau_n_after)
            criterion.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                data_Xn_val = Data(x=X_val[n], edge_index=edge_index)
                val_output = model.forward(data_Xn_val)
                val_loss = loss(val_output, s, X_val, n, tau_n_after).item()

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print(
                            f'Early stopping at epoch {epoch} as validation loss did not improve for {patience} consecutive epochs.')
                        break

            end_time = time.time()
            epoch_time = end_time - start_time
            times.append(epoch_time)
            losses.append(criterion.item())
        return F, model, losses, times

    elif model_type == "CNN":
        model = CNN1D(s.d, output_dim=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            start_time = time.time()
            # Training
            model.train()
            F = model.forward(X_train[n].unsqueeze(1))
            optimizer.zero_grad()
            criterion = loss(F, s, X_train, n, tau_n_after)
            criterion.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model.forward(X_val[n].unsqueeze(1))
                val_loss = loss(val_output, s, X_val, n, tau_n_after).item()

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print(
                            f'Early stopping at epoch {epoch} as validation loss did not improve for {patience} consecutive epochs.')
                        break

            end_time = time.time()
            epoch_time = end_time - start_time
            times.append(epoch_time)
            losses.append(criterion.item())
        return F, model, losses, times


def train_NN(S, X, model_type, threshold=0.5, lr=0.0001, train_split=0.8):
    """
    all_losses: training losses (iterations) at each timestep with length: N
    all_times: training times at each time step
    """

    N, num_path, _ = X.shape
    N = N - 1
    mods = [None] * N
    num_path = int(num_path*train_split)
    tau_mat = np.zeros((N + 1, num_path))
    tau_mat[N, :] = N
    f_mat = np.zeros((N + 1, num_path))
    f_mat[N, :] = 1
    all_losses = []
    all_times = []

    for n in range(N - 1, -1, -1):
        probs, mod_temp, losses, times = train_NN_stopping_t(n, X, S, torch.from_numpy(tau_mat[n + 1]).float(),
                                                             model_type, lr)
        mods[n] = mod_temp
        all_losses.append(losses)
        all_times.append(times)
        np_probs = probs.detach().numpy().reshape(num_path)
        print(n, ":", np.min(np_probs), " , ", np.max(np_probs))
        f_mat[n, :] = (np_probs > threshold) * 1.0
        tau_mat[n, :] = np.argmax(f_mat, axis=0)
    return mods, all_losses, all_times


def test_value_info(S, Y, mods, model_type, threshold=0.5):
    N, num_path, _ = Y.shape
    N = N - 1
    # Y = torch.from_numpy(S.GBM()).float()
    tau_mat_test = np.zeros((N + 1, num_path))
    tau_mat_test[N, :] = N
    f_mat_test = np.zeros((N + 1, num_path))
    f_mat_test[N, :] = 1
    V_mat_test = np.zeros((N + 1, num_path))
    V_est_test = np.zeros(N + 1)

    all_loss_test = np.zeros((N, 1))
    all_time_test = np.zeros((N, 1))

    for m in range(0, num_path):
        V_mat_test[N, m] = S.g(N, m, Y)
    V_est_test[N] = np.mean(V_mat_test[N, :])

    for n in range(N-1, -1, -1):
        start_time = time.time()
        mod_curr = mods[n]
        if model_type == "GNN":
            edge_index = torch.tensor([
                [i, i + 1] for i in range(N - 1)
            ], dtype=torch.long).t().contiguous()
            data_Yn = Data(x=Y[n], edge_index=edge_index)
            probs = mod_curr(data_Yn)

        elif model_type == "CNN":
            probs = mod_curr(Y[n].unsqueeze(1))

        else:
            probs = mod_curr(Y[n])

        np_probs = probs.detach().numpy().reshape(num_path)
        f_mat_test[n, :] = (np_probs > threshold) * 1.0
        tau_mat_test[n, :] = np.argmax(f_mat_test, axis=0)

        end_time = time.time()
        all_time_test[n] = end_time - start_time

        all_loss_test[n] = loss(probs, S, Y, n, torch.from_numpy(tau_mat_test[n + 1]).float()).item()

        for m in range(0, num_path):
            V_mat_test[n, m] = np.exp((n - tau_mat_test[n, m]) * (-S.r * S.T / S.N)) * S.g(tau_mat_test[n, m], m, Y)
    if (V_mat_test[0] == 0).all():
        V_mat_test = V_mat_test[1:]

    V_est_test = np.mean(V_mat_test, axis=1)
    V_std_test = np.std(V_mat_test, axis=1)
    V_se_test = V_std_test / (np.sqrt(num_path))
    z = scipy.stats.norm.ppf(0.975)
    lower = V_est_test[0] - z * V_se_test[0]
    upper = V_est_test[0] + z * V_se_test[0]
    print(V_est_test[0])
    print(V_se_test[0])
    print(lower)
    print(upper)

    return V_est_test[0], lower, upper, all_loss_test, all_time_test
