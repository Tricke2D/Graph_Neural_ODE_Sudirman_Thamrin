# 🚦 Graph_Neural_ODE_Sudirman_Thamrin

### Continuous-Time Traffic Speed Forecasting dengan Graph Neural ODE — Studi Kasus Koridor Sudirman–Thamrin–Bundaran HI, Jakarta

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)]()
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-latest-3C2179)]()
[![torchdiffeq](https://img.shields.io/badge/torchdiffeq-ODE_Solver-orange)]()
[![Gradio](https://img.shields.io/badge/Gradio-Demo_App-FF7C00?logo=gradio&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## 📍 Studi Kasus: Kenapa Koridor Ini?

Sudirman–Thamrin bukan sekadar jalan protokol — ini **urat nadi lalu lintas Jakarta**, tempat ribuan kendaraan bertemu setiap hari dari arah Blok M, Senayan, hingga Bundaran HI dan Kota. Setiap pagi dan sore, kemacetan di sini tidak muncul serentak di satu titik: ia **merambat**. Satu simpang macet, lalu menjalar ke simpang berikutnya, membentuk pola kemacetan yang berpindah seiring waktu di sepanjang jaringan jalan.

Masalahnya, kebanyakan pendekatan prediksi lalu lintas konvensional (rata-rata historis, ARIMA per-ruas, bahkan LSTM) memperlakukan tiap ruas jalan **secara terpisah** — padahal kemacetan adalah fenomena **spasial**: kondisi satu ruas jalan sangat dipengaruhi ruas-ruas di sekitarnya. Model yang tidak memahami struktur jaringan jalan akan kesulitan menangkap bagaimana kemacetan menjalar dari satu titik ke titik lain.

**Pertanyaan yang coba dijawab proyek ini:**
> Bagaimana memprediksi kecepatan lalu lintas di setiap ruas jalan Sudirman–Thamrin secara akurat, dengan tetap memperhitungkan bagaimana kemacetan menjalar melalui struktur graf jaringan jalan — dan bisa diprediksi ke berbagai horizon waktu tanpa retrain model terpisah untuk tiap horizon?

Jawabannya: **Graph Neural ODE (GNN-ODE)** — pendekatan yang menggabungkan Graph Convolutional Network (menangkap struktur spasial jaringan jalan) dengan Neural ODE (memodelkan evolusi kondisi lalu lintas sebagai sistem dinamis kontinu, bukan langkah waktu diskrit).

---

## 🖥️ Demo Aplikasi

<p align="center">
  <img src="assets/images/demo_screenshot.png" alt="Screenshot aplikasi Gradio - Prediksi Kecepatan Lalu Lintas" width="800"/>
</p>

<p align="center"><i>Tampilan utama: peta jaringan jalan real-time (warna = kecepatan), trayektori prediksi kontinu, dan panel benchmarking terhadap LSTM & ARIMA.</i></p>

> 📌 Ganti `assets/images/demo_screenshot.png` dengan screenshot asli aplikasi kamu (bisa ambil dari tab "1️⃣ Prediksi Real-time" di app Gradio yang sudah deploy di Railway).

---

## ✨ Fitur Utama

- **Continuous-time forecasting** — satu forward pass model bisa diintegrasikan ke horizon waktu berapa pun (bukan cuma kelipatan 5 menit), berkat solver ODE (`torchdiffeq`, metode RK4).
- **Spatial propagation modeling** — GCNConv sebagai fungsi turunan `dh/dt`, menangkap bagaimana kondisi node dipengaruhi node tetangganya di graf jalan.
- **Multi-horizon evaluation** — prediksi 5, 10, dan 15 menit ke depan dievaluasi dari satu model yang sama.
- **Benchmarking menyeluruh** — dibandingkan head-to-head dengan LSTM (neural non-graph), ARIMA (statistik klasik), dan baseline rata-rata historis.
- **Breakdown peak vs off-peak** — evaluasi terpisah untuk jam sibuk (07–09, 16–19) vs non-sibuk, use-case paling krusial untuk deteksi kemacetan.
- **Validasi struktur spasial** — pembuktian eksplisit bahwa model belajar propagasi graf (bukan cuma pola waktu per-node) lewat perbandingan korelasi node bertetangga vs node jauh.
- **Interactive demo** — aplikasi Gradio dengan peta interaktif, trayektori kecepatan, kurva turunan (dv/dt), dan analisis residual.

---

## 📊 Hasil Eksperimen

Evaluasi pada test set (1 hari, data disimulasikan dan dikalibrasi ke pola makro TomTom Traffic Index Jakarta 2025):

### (a) Perbandingan Full-Network (seluruh 2.669 node)

| Model               | MAE (km/h) | RMSE (km/h) | Cakupan     |
|---------------------|------------|-------------|-------------|
| **GNN-ODE**          | **0.60** ⭐ | **0.78** ⭐  | Semua node  |
| LSTM                | 4.04       | 5.22        | Semua node  |
| Baseline (avg jam)  | 5.13       | 6.16        | Semua node  |

→ GNN-ODE unggul **88.3%** dibanding baseline rata-rata historis sederhana.

### (b) Perbandingan Adil / Apples-to-Apples (20 node sample, termasuk ARIMA)

| Model               | MAE (km/h) | Cakupan          |
|---------------------|------------|------------------|
| **ARIMA**           | **0.63** ⭐ | 20 node sample   |
| GNN-ODE             | 0.68       | 20 node sample   |
| LSTM                | 4.06       | 20 node sample   |
| Baseline (avg jam)  | 5.37       | 20 node sample   |

### (c) Evaluasi Multi-Horizon (satu model, integrasi ODE ke t berbeda)

| Horizon    | GNN-ODE MAE | LSTM MAE | ARIMA MAE |
|------------|-------------|----------|-----------|
| 5 menit    | 0.60        | 4.04     | 0.75      |
| 10 menit   | 1.06        | 4.10     | 0.82      |
| 15 menit   | 1.61        | 4.14     | 0.96      |

### (d) Breakdown Peak vs Off-Peak

| Kondisi     | MAE (km/h) | RMSE (km/h) | n   |
|-------------|------------|-------------|-----|
| Jam Sibuk   | 0.70       | 0.89        | 84  |
| Non-Sibuk   | 0.55       | 0.72        | 192 |

> ⚠️ **Catatan jujur:** angka MAE yang sangat rendah ini adalah prediksi **1-step (5 menit ke depan)** dengan autokorelasi tinggi terhadap kondisi saat ini, pada data **sintetis** (bukan sensor riil), dan test set hanya mencakup 1 hari. Lihat tab "Batasan Model" di aplikasi untuk konteks lengkap sebelum menafsirkan angka ini sebagai performa absolut.

---

## 🏗️ Arsitektur Model

```
Input (12 langkah historis, 1 jam)
        │
        ▼
   Encoder (Linear)
        │
        ▼
  ┌─────────────────┐
  │   ODEFunc        │   dh/dt = GCN(h)
  │  GCNConv → ReLU  │   diintegrasikan via
  │  → GCNConv       │   torchdiffeq.odeint (RK4)
  └─────────────────┘
        │
        ▼
   Decoder (Linear)
        │
        ▼
  Prediksi kecepatan (t + horizon)
```

Karena integrasi dilakukan di ruang waktu kontinu, model bisa dievaluasi pada **titik waktu manapun** (t=5, t=7.3, t=15 menit, dst) dari satu forward pass yang sama — inilah yang membedakan GNN-ODE dari model diskrit seperti LSTM/GCN biasa.

---

## 🛠️ Tech Stack

- Python 3.9+
- Jupyter Notebook / Google Colab
- PyTorch
- PyTorch Geometric (GCNConv)
- torchdiffeq (Neural ODE solver)
- OSMnx & NetworkX (road network graph)
- Pandas & NumPy
- Matplotlib / Plotly
- statsmodels (ARIMA benchmark)
- Gradio (interactive demo)

---

## 💻 Requirements

- CPU: Multi-core processor (recommended); GPU opsional, mempercepat training tapi tidak wajib untuk inferensi
- RAM: Minimum 8 GB (16 GB recommended untuk training penuh 2.669 node)
- Python >= 3.9

---

## 🚀 Instalasi & Menjalankan

```bash
git clone https://github.com/username/Graph_Neural_ODE_Sudirman_Thamrin.git
cd Graph_Neural_ODE_Sudirman_Thamrin
pip install -r requirements.txt
```

**Menjalankan notebook riset** (urutan wajib, tiap flow menyimpan artefak untuk flow berikutnya):
```
notebooks/01_data_collection_preprocessing.ipynb
notebooks/02_eda_validation.ipynb
notebooks/03_model_architecture.ipynb
notebooks/04_training_evaluation.ipynb
notebooks/05_deployment_gradio.ipynb
```

**Menjalankan demo aplikasi secara lokal:**
```bash
cd app
python app.py
```

---

## 📁 Struktur Repository

```
Graph_Neural_ODE_Sudirman_Thamrin/
├── README.md
├── requirements.txt
├── notebooks/          # 5 notebook riset, urut sesuai flow
├── src/                # modul Python hasil ekstraksi (data, models, utils)
├── assets/images/      # screenshot demo, chart hasil eksperimen
├── results/            # tabel metrik hasil evaluasi
├── app/                # aplikasi Gradio (deployment terpisah)
└── docs/               # dokumentasi lengkap (PDF)
```

---

## ⚠️ Batasan & Pengembangan Selanjutnya

- Data kecepatan bersifat **sintetis**, dikalibrasi ke pola makro TomTom Traffic Index Jakarta 2025 — struktur graf dari OSM asli, tapi nilai kecepatan adalah simulasi terkontrol, bukan observasi sensor riil per-ruas.
- Test set hanya mencakup **1 hari** — breakdown performa bersifat indikatif, belum tentu generalisasi ke semua kondisi.
- ARIMA hanya dievaluasi pada **sample 20 node** karena biaya komputasi fit-per-node.
- Rencana pengembangan: integrasi data sensor riil (mis. dari GPS kendaraan umum atau Google/TomTom API), perluasan area cakupan, dan pipeline real-time.

---

## 👤 Author

**[Nama Kamu]**
Informatika, Universitas AMIKOM Yogyakarta
[LinkedIn](#) · [GitHub](#)

## 📄 License

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
