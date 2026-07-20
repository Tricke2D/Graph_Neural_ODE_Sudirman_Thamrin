"""
Gradio Demo — Prediksi Kecepatan Lalu Lintas Berbasis Graph Neural ODE
Koridor Sudirman-Thamrin-Bundaran HI

Cara pakai:
1. Jalankan cell export "Flow 5" di notebook (Graph_Neural_ODE_Sudirman_Thamrin_v5.ipynb)
   sampai selesai -> hasilkan folder `gradio_artifacts/`.
2. Taruh folder itu satu direktori dengan file ini.
3. pip install -r requirements.txt
4. python app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
import gradio as gr

try:
    from torch_geometric.nn import GCNConv
    from torchdiffeq import odeint
except ImportError as e:
    raise SystemExit(
        "Dependency inti belum terpasang. Jalankan: pip install -r requirements.txt\n"
        f"Detail error: {e}"
    )

ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradio_artifacts")
STEP_MIN = 5  # resolusi asli data (menit per step)
VALIDATED_MAX_HORIZON_MIN = 15  # horizon terjauh yang benar-benar dievaluasi di notebook

# ============================================================
# 1. LOAD ARTEFAK
# ============================================================
if not os.path.isdir(ART_DIR):
    raise SystemExit(
        f"Folder artefak tidak ditemukan: {ART_DIR}\n"
        "Jalankan dulu cell export 'Flow 5' di notebook, lalu salin folder "
        "gradio_artifacts/ ke direktori yang sama dengan app.py ini."
    )

with open(os.path.join(ART_DIR, "bundle.pkl"), "rb") as f:
    B = pickle.load(f)

config = B["config"]
train_mean, train_std = B["train_mean"], B["train_std"]
node_lat, node_lon = B["node_lat"], B["node_lon"]
edge_list_idx = B["edge_list_idx"]
examples = B["examples"]
metrics_table = B["metrics_table"]
horizon_analysis = B["horizon_analysis"]
residuals = B["residuals"]
residual_actual_pairs = B["residual_actual_pairs"]
overlay = B["overlay"]
global_note = B["global_rmse_note"]

device = torch.device("cpu")
edge_index = torch.load(os.path.join(ART_DIR, "edge_index.pt")).to(device)
NUM_NODES = config["num_nodes"]


# ============================================================
# 2. DEFINISI MODEL (harus identik dengan notebook)
# ============================================================
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, edge_index):
        super().__init__()
        self.edge_index = edge_index
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, t, h):
        out = F.relu(self.gcn1(h, self.edge_index))
        out = self.gcn2(out, self.edge_index)
        return out


class GNN_ODE(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_index, solver="rk4"):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim, edge_index)
        self.decoder = nn.Linear(hidden_dim, 1)
        self.solver = solver

    def encode_h0(self, x):
        h0 = x[:, -1, :].unsqueeze(-1)
        return self.encoder(h0)


class LSTMForecast(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(num_nodes, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.decoder(out[:, -1, :])


gnn_model = GNN_ODE(config["in_dim"], config["hidden_dim"], edge_index, config["solver"]).to(device)
gnn_model.load_state_dict(torch.load(os.path.join(ART_DIR, "best_gnn_ode_model.pt"), map_location=device))
gnn_model.eval()

lstm_model = LSTMForecast(NUM_NODES, hidden_dim=64).to(device)
lstm_model.load_state_dict(torch.load(os.path.join(ART_DIR, "best_lstm_model.pt"), map_location=device))
lstm_model.eval()


# ============================================================
# 3. UTILITAS PREDIKSI
# ============================================================
def normalize(x_raw):
    return (x_raw - train_mean) / train_std


def denorm(x_norm):
    return x_norm * train_std + train_mean


@torch.no_grad()
def gnn_ode_dense_trajectory(x_seq_raw, horizon_min, n_points=40):
    """Integrasikan GNN-ODE dari t=0 s/d horizon_min pada grid waktu padat.
    Inilah yang membedakan GNN-ODE dari model diskrit: satu forward pass bisa
    dievaluasi di titik waktu manapun (bukan cuma kelipatan 5 menit)."""
    horizon_steps = max(horizon_min, 0.1) / STEP_MIN
    t_grid = torch.linspace(0.0, horizon_steps, n_points)
    x = torch.tensor(normalize(x_seq_raw), dtype=torch.float32).unsqueeze(0)
    h0 = gnn_model.encode_h0(x)[0]  # [num_nodes, hidden_dim]
    h_traj = odeint(gnn_model.odefunc, h0, t_grid, method=gnn_model.solver)  # [n_points, num_nodes, hidden]
    pred_traj = gnn_model.decoder(h_traj).squeeze(-1)  # [n_points, num_nodes]
    pred_kph = denorm(pred_traj.numpy())
    t_minutes = (t_grid.numpy() * STEP_MIN)
    return t_minutes, pred_kph  # [n_points], [n_points, num_nodes]


@torch.no_grad()
def lstm_rollout(x_seq_raw, horizon_min):
    """LSTM tidak kontinu -- rollout autoregresif per step 5 menit sampai horizon."""
    horizon_steps = max(1, round(horizon_min / STEP_MIN))
    x = torch.tensor(normalize(x_seq_raw), dtype=torch.float32).unsqueeze(0)
    cur_seq = x.clone()
    last_pred = None
    for _ in range(horizon_steps):
        last_pred = lstm_model(cur_seq)
        cur_seq = torch.cat([cur_seq[:, 1:, :], last_pred.unsqueeze(1)], dim=1)
    return denorm(last_pred.numpy()[0])  # [num_nodes]


# ============================================================
# 4. VISUALISASI
# ============================================================
COLORSCALE = "RdYlGn"
SPEED_MIN, SPEED_MAX = 5, 80

# Precompute koordinat garis edge sekali saja (tidak berubah antar prediksi)
_EDGE_X, _EDGE_Y = [], []
for (ui, vi) in edge_list_idx:
    _EDGE_X += [float(node_lon[ui]), float(node_lon[vi]), None]
    _EDGE_Y += [float(node_lat[ui]), float(node_lat[vi]), None]


def build_network_figure(speed_values, title):
    edge_trace = go.Scattergl(
        x=_EDGE_X, y=_EDGE_Y, mode="lines",
        line=dict(width=1, color="rgba(140,140,140,0.35)"),
        hoverinfo="none", showlegend=False,
    )
    node_trace = go.Scattergl(
        x=node_lon, y=node_lat, mode="markers",
        marker=dict(
            size=5, color=speed_values, colorscale=COLORSCALE,
            cmin=SPEED_MIN, cmax=SPEED_MAX,
            colorbar=dict(title="km/h"),
        ),
        text=[f"Node {i}: {s:.1f} km/h" for i, s in enumerate(speed_values)],
        hoverinfo="text", showlegend=False,
    )
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=40, b=0), height=480,
        plot_bgcolor="white",
    )
    return fig


def build_trajectory_figure(t_minutes, pred_kph, node_idx):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_minutes, y=pred_kph[:, node_idx],
        mode="lines", line=dict(color="#1f77b4", width=3),
        name=f"Node {node_idx}",
    ))
    fig.update_layout(
        title=f"Trayektori Kecepatan Kontinu — Node {node_idx}",
        xaxis_title="Menit ke depan", yaxis_title="Kecepatan (km/h)",
        height=340, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_derivative_figure(t_minutes, pred_kph, node_idx):
    v = pred_kph[:, node_idx]
    dv_dt = np.gradient(v, t_minutes)  # km/h per menit
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_trace(go.Scatter(
        x=t_minutes, y=dv_dt, mode="lines", line=dict(color="#d62728", width=2.5),
        name="dv/dt",
    ))
    fig.update_layout(
        title=f"Laju Perubahan Kecepatan (dv/dt) — Node {node_idx}",
        xaxis_title="Menit ke depan", yaxis_title="dv/dt (km/h per menit)",
        height=300, margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_overlay_figure():
    ts = pd.to_datetime(overlay["timestamps"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=overlay["actual"], mode="lines", name="Aktual",
                              line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=ts, y=overlay["gnn_ode"], mode="lines", name="GNN-ODE",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=ts, y=overlay["lstm"], mode="lines", name="LSTM",
                              line=dict(color="#ff7f0e", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=ts, y=overlay["arima"], mode="lines", name="ARIMA",
                              line=dict(color="#2ca02c", width=2, dash="dot")))
    fig.update_layout(
        title=f"Aktual vs Prediksi 3 Model — Node {overlay['node_idx']} (Test Set, 1 Hari)",
        xaxis_title="Waktu", yaxis_title="Kecepatan (km/h)",
        height=420, legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_horizon_error_figure():
    h = horizon_analysis
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["horizons_min"], y=h["gnn_ode_mae"], mode="lines+markers",
                              name="GNN-ODE", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=h["horizons_min"], y=h["lstm_mae"], mode="lines+markers",
                              name="LSTM", line=dict(color="#ff7f0e")))
    fig.add_trace(go.Scatter(x=h["horizons_min"], y=h["arima_mae"], mode="lines+markers",
                              name="ARIMA", line=dict(color="#2ca02c")))
    fig.update_layout(
        title="MAE vs Panjang Horizon Prediksi (20 node sample)",
        xaxis_title="Horizon (menit)", yaxis_title="MAE (km/h)",
        height=380, legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_residual_figure():
    fig = go.Figure()
    for name, color in [("gnn_ode", "#1f77b4"), ("lstm", "#ff7f0e"), ("arima", "#2ca02c")]:
        fig.add_trace(go.Histogram(x=residuals[name], name=name.upper(), opacity=0.55,
                                    marker_color=color, nbinsx=60))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Distribusi Residual (Prediksi - Aktual)", barmode="overlay",
        xaxis_title="Residual (km/h)", yaxis_title="Frekuensi",
        height=380, legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_residual_vs_actual_figure():
    fig = go.Figure()
    for name, color in [("gnn_ode", "#1f77b4"), ("lstm", "#ff7f0e"), ("arima", "#2ca02c")]:
        idx = np.random.RandomState(0).choice(len(residuals[name]), size=min(3000, len(residuals[name])), replace=False)
        fig.add_trace(go.Scattergl(
            x=residual_actual_pairs[name][idx], y=residuals[name][idx],
            mode="markers", name=name.upper(), marker=dict(size=4, color=color, opacity=0.4),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Residual vs Kecepatan Aktual (cek bias sistematis)",
        xaxis_title="Kecepatan aktual (km/h)", yaxis_title="Residual (km/h)",
        height=380, legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def build_metrics_markdown():
    df_full = pd.DataFrame(metrics_table["full"])
    df_sample = pd.DataFrame(metrics_table["sample"])

    def star_best(df, col):
        best_idx = df[col].idxmin()
        out = []
        for i, v in enumerate(df[col]):
            cell = f"**{v:.2f}**" if i == best_idx else f"{v:.2f}"
            if i == best_idx:
                cell += " ⭐"
            out.append(cell)
        return out

    df_full = df_full.copy()
    df_full["MAE (km/h)"] = star_best(df_full, "MAE (km/h)")
    df_full["RMSE (km/h)"] = star_best(df_full, "RMSE (km/h)")

    df_sample = df_sample.copy()
    df_sample["MAE (km/h)"] = star_best(df_sample, "MAE (km/h)")

    md = "#### (a) Perbandingan Full-Network (semua node; ARIMA tidak diikutkan — hanya di-sample)\n\n"
    md += df_full.to_markdown(index=False) + "\n\n"
    md += "#### (b) Perbandingan Adil / Apples-to-Apples (20 node sample, keempat model)\n\n"
    md += df_sample.to_markdown(index=False)
    return md


# ============================================================
# 5. CALLBACK UTAMA
# ============================================================
def run_prediction(scenario_name, horizon_min, node_idx, mode):
    try:
        node_idx = int(node_idx)
        if not (0 <= node_idx < NUM_NODES):
            return (gr.update(), gr.update(), gr.update(),
                    f"⚠️ Node index harus di antara 0 dan {NUM_NODES - 1}.")

        x_seq_raw = examples[scenario_name]  # [seq_len, num_nodes]

        t_minutes, pred_traj_kph = gnn_ode_dense_trajectory(x_seq_raw, horizon_min, n_points=40)
        speed_now_pred = pred_traj_kph[-1]  # kondisi di horizon yang diminta -> untuk peta

        net_fig = build_network_figure(speed_now_pred, f"Prediksi GNN-ODE — t+{horizon_min:.0f} menit")
        traj_fig = build_trajectory_figure(t_minutes, pred_traj_kph, node_idx)
        deriv_fig = build_derivative_figure(t_minutes, pred_traj_kph, node_idx)

        lstm_speed_node = lstm_rollout(x_seq_raw, horizon_min)[node_idx]
        gnn_speed_node = speed_now_pred[node_idx]

        warn = ""
        if mode == "Mode Long-term (Multi-step)" and horizon_min > VALIDATED_MAX_HORIZON_MIN:
            warn = (
                f"\n\n⚠️ **Peringatan ekstrapolasi:** horizon {horizon_min:.0f} menit melewati "
                f"batas yang dievaluasi di notebook (maks {VALIDATED_MAX_HORIZON_MIN} menit). "
                "Akurasi di luar rentang ini tidak terverifikasi — lihat tab 'Batasan Model'."
            )

        status = (
            f"✅ Prediksi selesai — skenario **{scenario_name}**, horizon **{horizon_min:.0f} menit**.\n\n"
            f"- Node {node_idx}: GNN-ODE = **{gnn_speed_node:.1f} km/h**, LSTM (rollout) = **{lstm_speed_node:.1f} km/h**\n"
            f"- MAE 1-step GNN-ODE (test set): **{global_note['gnn_ode_mae_1step']:.2f} km/h** "
            f"(_prediksi jangka sangat pendek, autokorelasi tinggi — lihat tab Batasan Model_)"
            + warn
        )
        return net_fig, traj_fig, deriv_fig, status
    except Exception as e:
        return gr.update(), gr.update(), gr.update(), f"❌ Terjadi error: {e}"


# ============================================================
# 6. LAYOUT GRADIO
# ============================================================
LIMITATION_TEXT = f"""
## ⚠️ Batasan Model — Wajib Dibaca Sebelum Menafsirkan Hasil

**RMSE 1-step GNN-ODE pada test set: {global_note['gnn_ode_rmse_1step']:.2f} km/h.**
Angka ini **sangat rendah** dan mudah disalahtafsirkan sebagai "model hampir sempurna".
Berikut konteks yang perlu diperhatikan:

1. **Ini prediksi 1-step (5 menit ke depan), bukan multi-step jangka panjang.**
   Prediksi jangka sangat pendek punya autokorelasi tinggi terhadap kondisi saat ini —
   RMSE akan naik signifikan pada horizon yang lebih jauh (lihat tab Benchmarking →
   Error vs Horizon).

2. **Data kecepatan bersifat sintetis**, digenerate lewat fungsi `congestion_factor()`
   yang dikalibrasi ke pola makro TomTom Traffic Index Jakarta 2025 — bukan observasi
   sensor riil per-ruas-jalan. Struktur graf (node/edge) memang dari OSM asli, tapi
   nilai kecepatannya adalah simulasi terkontrol.

3. **Test set hanya mencakup 1 hari.** Breakdown performa (jam sibuk vs non-sibuk,
   apples-to-apples vs ARIMA) bersifat indikatif, bukan kesimpulan final yang
   generalisasi ke semua kondisi.

4. **ARIMA hanya dievaluasi pada sample 20 node** (bukan seluruh jaringan) karena biaya
   komputasi fit-per-node. Perbandingan "adil" di tab Benchmarking menggunakan 20 node
   yang sama untuk keempat model.

5. **Horizon tervalidasi hingga {VALIDATED_MAX_HORIZON_MIN} menit.** Slider di sidebar
   mengizinkan horizon lebih panjang untuk eksplorasi kemampuan kontinu ODE, tapi
   akurasinya di luar rentang ini belum diverifikasi terhadap ground truth.

**Kesimpulan:** aplikasi ini adalah **alat demonstrasi ilmiah** tentang bagaimana
GNN-ODE bekerja dan bagaimana ia dibandingkan dengan LSTM/ARIMA pada skenario
terkalibrasi — bukan klaim performa absolut yang siap produksi.
"""

with gr.Blocks(title="GNN-ODE Traffic Speed Prediction") as demo:
    gr.Markdown(
        "# 🚦 Prediksi Kecepatan Lalu Lintas Berbasis Graf & ODE\n"
        "Koridor **Sudirman–Thamrin–Bundaran HI**, Jakarta. Model: **Graph Neural Network "
        "+ Neural ODE (GNN-ODE)**, dibandingkan dengan LSTM dan ARIMA. Struktur graf jalan "
        "dari data OSM asli; nilai kecepatan disimulasikan (lihat tab *Batasan Model*)."
    )

    with gr.Row():
        # ---------------- SIDEBAR ----------------
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Panel Input")
            scenario_dd = gr.Dropdown(
                choices=list(examples.keys()), value=list(examples.keys())[0],
                label="Skenario Contoh (Fitur Dinamis: 1 jam historis)",
            )
            mode_radio = gr.Radio(
                choices=["Mode Short-term (1-step)", "Mode Long-term (Multi-step)"],
                value="Mode Short-term (1-step)",
                label="Saklar Mode Prediksi",
            )
            horizon_slider = gr.Slider(
                minimum=5, maximum=60, step=5, value=5,
                label="Horizon Prediksi (menit ke depan)",
            )
            node_input = gr.Number(
                value=int(np.argmin(np.abs(node_lat - node_lat.mean()))),
                label=f"Index Node untuk Detail Trayektori (0–{NUM_NODES - 1})",
                precision=0,
            )
            predict_btn = gr.Button("🔮 Prediksi & Bandingkan", variant="primary")
            status_md = gr.Markdown("Status: menunggu input pengguna.")
            gr.Markdown(
                f"*Fitur Statis: struktur graf jalan ({NUM_NODES} node, "
                f"{len(edge_list_idx)} ruas) dari OpenStreetMap.*"
            )

        # ---------------- MAIN PANEL ----------------
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("1️⃣ Prediksi Real-time"):
                    network_plot = gr.Plot(label="Peta Jaringan Jalan (warna = kecepatan)")
                    with gr.Row():
                        traj_plot = gr.Plot(label="Trayektori Kontinu")
                        deriv_plot = gr.Plot(label="Turunan (dv/dt)")

                with gr.Tab("2️⃣ Benchmarking (GNN-ODE vs LSTM vs ARIMA)"):
                    gr.Markdown(build_metrics_markdown())
                    overlay_plot = gr.Plot(value=build_overlay_figure(),
                                            label="Overlay Aktual vs 3 Model")
                    horizon_plot = gr.Plot(value=build_horizon_error_figure(),
                                            label="Error vs Horizon")

                with gr.Tab("3️⃣ Analisis Error"):
                    residual_hist_plot = gr.Plot(value=build_residual_figure(),
                                                  label="Histogram Residual")
                    residual_scatter_plot = gr.Plot(value=build_residual_vs_actual_figure(),
                                                     label="Residual vs Aktual")

                with gr.Tab("⚠️ Batasan Model"):
                    gr.Markdown(LIMITATION_TEXT)

            gr.Markdown("---")
            df_full_footer = pd.DataFrame(metrics_table["full"])
            footer_txt = "**Ringkasan Metrik Global:** " + " | ".join(
                f"{r['Model']}: MAE {r['MAE (km/h)']:.2f} / RMSE {r['RMSE (km/h)']:.2f} km/h"
                for r in df_full_footer.to_dict(orient="records")
            )
            gr.Markdown(footer_txt)

    predict_btn.click(
        fn=run_prediction,
        inputs=[scenario_dd, horizon_slider, node_input, mode_radio],
        outputs=[network_plot, traj_plot, deriv_plot, status_md],
    )
    demo.load(
        fn=run_prediction,
        inputs=[scenario_dd, horizon_slider, node_input, mode_radio],
        outputs=[network_plot, traj_plot, deriv_plot, status_md],
    )

if __name__ == "__main__":
       demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
