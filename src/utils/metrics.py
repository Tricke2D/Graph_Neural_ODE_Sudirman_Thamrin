"""
Utilitas metrik evaluasi model forecasting.
Sumber: notebooks/04_training_evaluation.ipynb (Flow 4, evaluasi test set & benchmarking)

Dipakai berulang di beberapa tahap: evaluasi test set utama, evaluasi
multi-horizon, breakdown peak vs off-peak, dan perbandingan GNN-ODE vs
LSTM vs ARIMA vs baseline historis.
"""

import numpy as np


def mae(preds, targets):
    """Mean Absolute Error."""
    return np.mean(np.abs(preds - targets))


def rmse(preds, targets):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((preds - targets) ** 2))


def denormalize(x_norm, mean, std):
    """Kembalikan nilai ternormalisasi ke satuan asli (km/h).
    mean & std WAJIB dihitung hanya dari data train (hindari data leakage)."""
    return x_norm * std + mean


def normalize(x_raw, mean, std):
    """Normalisasi nilai mentah pakai mean & std dari data train."""
    return (x_raw - mean) / std


def is_peak_hour(hour):
    """Definisi jam sibuk: 07-09 (pagi) dan 16-19 (sore)."""
    return (7 <= hour <= 9) or (16 <= hour <= 19)


def evaluate_peak_vs_offpeak(preds, targets, hours):
    """Breakdown MAE/RMSE untuk jam sibuk vs non-sibuk.

    preds, targets : array [n_samples, n_nodes] dalam km/h
    hours          : array [n_samples] berisi jam (0-23) tiap sample
    """
    peak_mask = np.array([is_peak_hour(h) for h in hours])

    return {
        "mae_peak": mae(preds[peak_mask], targets[peak_mask]),
        "mae_offpeak": mae(preds[~peak_mask], targets[~peak_mask]),
        "rmse_peak": rmse(preds[peak_mask], targets[peak_mask]),
        "rmse_offpeak": rmse(preds[~peak_mask], targets[~peak_mask]),
    }


def summarize_model_comparison(model_metrics: dict):
    """model_metrics: {"GNN-ODE": {"mae": ..., "rmse": ...}, "LSTM": {...}, ...}
    Return dict yang sama plus penanda model terbaik per metrik (MAE terendah)."""
    best_model = min(model_metrics, key=lambda m: model_metrics[m]["mae"])
    return {"metrics": model_metrics, "best_by_mae": best_model}
