"""
ODEFunc — fungsi turunan f_theta(h, t) yang diintegrasikan lewat torchdiffeq.odeint().

Definisi ini HARUS identik dengan yang dipakai di app.py (Gradio) dan notebook
04_training_evaluation.ipynb, karena PyTorch butuh arsitektur class yang sama
persis untuk bisa memuat kembali state_dict (.pt) yang sudah dilatih.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ODEFunc(nn.Module):
    """f_theta(h, t) = GCN(h) -- ini yang diintegrasikan odeint.

    2 layer GCNConv menangkap struktur spasial graf jalan: kondisi node
    dipengaruhi oleh node tetangganya lewat message passing, lalu hasil ini
    dipakai solver ODE untuk mengintegrasikan hidden state secara kontinu
    dari t=0 ke t=target.
    """

    def __init__(self, hidden_dim, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, t, h):
        out = F.relu(self.gcn1(h, self.edge_index))
        out = self.gcn2(out, self.edge_index)
        return out
