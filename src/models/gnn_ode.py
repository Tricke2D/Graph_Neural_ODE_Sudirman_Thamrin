"""
GNN_ODE — model utama: Encoder (Linear) -> ODEFunc (GCN, via odeint) -> Decoder (Linear).

Alur:
1. encode_h0(x)  : proyeksikan speed langkah terakhir tiap node ke hidden_dim.
2. odeint(...)   : integrasikan hidden state secara kontinu dari t=0 ke t=target
                   (dilakukan di luar class ini, lihat notebook 04 / app.py,
                   karena butuh torchdiffeq.odeint langsung).
3. decoder(h)    : kembalikan hidden state ke prediksi kecepatan (1 nilai per node).

Keunggulan dibanding GCN diskrit biasa: karena solver ODE bekerja pada domain
waktu kontinu, satu forward pass bisa dievaluasi di horizon berapa pun
(5 menit, 12.5 menit, dst) tanpa retrain ulang untuk tiap horizon.
"""

import torch.nn as nn

from .ode_func import ODEFunc


class GNN_ODE(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_index, solver="rk4"):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim, edge_index)
        self.decoder = nn.Linear(hidden_dim, 1)
        self.solver = solver

    def encode_h0(self, x):
        """x: [batch, seq_len, num_nodes] -> ambil langkah waktu terakhir,
        proyeksikan ke hidden_dim. Return: [batch, num_nodes, hidden_dim]."""
        h0 = x[:, -1, :].unsqueeze(-1)
        return self.encoder(h0)
