"""
LSTMForecast — baseline neural non-graph.

Memprediksi kecepatan semua node sekaligus dari sequence historis, TANPA
memanfaatkan struktur graf (edge_index) sama sekali. Dipakai sebagai
pembanding untuk membuktikan bahwa keunggulan GNN-ODE (jika ada) memang
berasal dari pemodelan struktur spasial graf, bukan cuma dari kapasitas
neural network secara umum.
"""

import torch.nn as nn


class LSTMForecast(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(num_nodes, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.decoder(out[:, -1, :])
