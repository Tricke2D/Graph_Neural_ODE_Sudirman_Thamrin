"""
Generator data kecepatan lalu lintas sintetis.
Sumber: notebooks/01_data_collection_preprocessing.ipynb (Flow 1, cell 1.6)

Karena data sensor riil per-ruas-jalan tidak tersedia publik, kecepatan
disimulasikan lewat congestion_factor() yang dikalibrasi ke pola makro
TomTom Traffic Index Jakarta (data 2025, rilis Jan 2026):
  - Congestion Level (CL) jam macet pagi (07.00-09.00 WIB) ~= 43%
    -> speed_mult ~= 0.70  [sumber: traveloka.com, kutip TomTom]
  - CL jam macet sore (16.30-19.00 WIB, puncak ~17.45) ~= 62-67%
    -> speed_mult ~= 0.60  [sumber: traveloka.com; viva.co.id Okt 2025]
  - CL rata-rata tahunan 2025 = 59.8%, kecepatan rata-rata jam sibuk = 22.8 km/h
    [sumber: tomtom.com/traffic-index/city/jakarta]

Struktur graf (node/edge) berasal dari OSM asli; nilai kecepatannya adalah
simulasi terkontrol -- lihat tab "Batasan Model" di app.py untuk konteks ini.
"""

import numpy as np
import pandas as pd


def congestion_factor(ts):
    """Return faktor pengali speed (0.15 - 1.0) berdasarkan jam & hari."""
    hour = ts.hour + ts.minute / 60
    is_weekend = ts.dayofweek >= 5

    if is_weekend:
        base = 0.85
        dip = 0.15 * np.exp(-((hour - 14) ** 2) / 8)
    else:
        base = 0.9
        dip_pagi = 0.20 * np.exp(-((hour - 8.0) ** 2) / 1.2)
        dip_sore = 0.30 * np.exp(-((hour - 17.75) ** 2) / 2.2)
        dip = dip_pagi + dip_sore

    factor = base - dip
    return np.clip(factor, 0.15, 1.0)


def generate_synthetic_speed(G, start="2026-01-06 00:00", days=7, freq="5min", seed=42):
    """Generate data kecepatan sintetis per-edge untuk seluruh rentang waktu.

    G harus punya atribut edge 'speed_kph' (free-flow speed), hasil default
    OSMnx `add_edge_speeds()`.
    """
    np.random.seed(seed)

    start_ts = pd.Timestamp(start)
    end_ts = start_ts + pd.Timedelta(days=days)
    timestamps = pd.date_range(start_ts, end_ts, freq=freq)[:-1]

    time_factors = np.array([congestion_factor(ts) for ts in timestamps])

    edges = list(G.edges(keys=True, data=True))
    n_time = len(timestamps)

    node_ids = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(node_ids)}

    # Random walk noise per node biar smooth antar waktu, dan edge yang
    # berbagi node akan punya noise yang mirip (korelasi spasial sederhana).
    node_noise = np.zeros((len(node_ids), n_time))
    for i in range(len(node_ids)):
        walk = np.cumsum(np.random.normal(0, 0.02, n_time))
        walk = walk - walk.mean()
        node_noise[i] = np.clip(walk, -0.25, 0.25)

    records = []
    for (u, v, k, data) in edges:
        free_flow = data["speed_kph"]
        ui, vi = node_idx[u], node_idx[v]
        edge_noise = (node_noise[ui] + node_noise[vi]) / 2
        factor = np.clip(time_factors + edge_noise, 0.1, 1.0)
        speed = free_flow * factor
        speed += np.random.normal(0, 1.0, n_time)  # sensor noise kecil
        speed = np.clip(speed, 3, free_flow)

        for t_idx, ts in enumerate(timestamps):
            records.append((u, v, k, ts, speed[t_idx]))

    df_speed = pd.DataFrame(records, columns=["u", "v", "k", "timestamp", "speed_kph"])
    print(f"Generated {len(df_speed):,} rows untuk {len(edges)} edges x {n_time} timestamps")
    return df_speed


if __name__ == "__main__":
    import pickle

    with open("sudirman_thamrin_graph_final.pkl", "rb") as f:
        G = pickle.load(f)

    df_speed = generate_synthetic_speed(G)
    df_speed.to_parquet("synthetic_traffic_speed.parquet", index=False)
