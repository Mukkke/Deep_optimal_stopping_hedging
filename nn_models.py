import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class mlp_stopping(torch.nn.Module):
    def __init__(self, d, q1, q2):
        super(mlp_stopping, self).__init__()
        self.a1 = nn.Linear(d, q1)
        self.leaky_relu = F.leaky_relu
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.a1.weight)
        self.a2 = nn.Linear(q1, q2)
        nn.init.xavier_uniform_(self.a2.weight)
        self.a3 = nn.Linear(q2, 1)
        nn.init.xavier_uniform_(self.a3.weight)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(q1)
        self.bn2 = nn.BatchNorm1d(q2)

    def forward(self, x):
        out = self.a1(x)
        out = self.leaky_relu(out)
        # out = self.relu(out)
        out = self.bn1(out) #####
        out = self.dropout(out)
        out = self.a2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.a3(out)
        out = self.sigmoid(out)
        return out


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.sigmoid(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class CNN1D(nn.Module):
    def __init__(self, input_size, output_dim=1, hidden_dim1=16, hidden_dim2=32):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim1, out_channels=hidden_dim2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
