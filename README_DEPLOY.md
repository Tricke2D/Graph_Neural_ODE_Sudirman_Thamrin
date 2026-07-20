# Cara Deploy Demo Gradio — GNN-ODE Traffic Speed

## Alur singkat
1. Buka **Graph_Neural_ODE_Sudirman_Thamrin_v5.ipynb** (sudah ditambah 3 cell baru di
   paling bawah: header "Flow 5", cell export artefak, cell zip+download).
2. Jalankan seluruh notebook seperti biasa dari atas ke bawah (tidak ada cell lama yang
   diubah), lalu jalankan 3 cell baru di bagian **Flow 5** paling akhir.
3. Cell export akan membuat folder `gradio_artifacts/` berisi:
   - `bundle.pkl` — koordinat graf, skenario contoh, tabel metrik, hasil error-vs-horizon,
     residual, data overlay
   - `best_gnn_ode_model.pt`, `best_lstm_model.pt` — bobot model (untuk inferensi live)
   - `edge_index.pt` — struktur graf untuk GCNConv
4. Cell terakhir men-zip folder itu jadi `gradio_artifacts.zip` dan otomatis
   men-trigger download di Colab.
5. Di komputer/server tujuan deploy:
   ```
   mkdir gnn_ode_demo && cd gnn_ode_demo
   unzip gradio_artifacts.zip -d gradio_artifacts
   # taruh app.py dan requirements.txt di folder gnn_ode_demo/ (sejajar dgn gradio_artifacts/)
   pip install -r requirements.txt
   python app.py
   ```
6. Gradio akan berjalan di `http://127.0.0.1:7860`. Untuk deploy publik: HuggingFace
   Spaces (upload `app.py`, `requirements.txt`, folder `gradio_artifacts/`) atau server
   sendiri di belakang reverse proxy.

## Catatan penting
- ARIMA **tidak** dijalankan live di app.py (fit-per-node terlalu lambat untuk UI
  interaktif). Nilai ARIMA di tab Benchmarking berasal dari hasil pra-hitung di notebook,
  konsisten dengan disclaimer "sample 20 node" yang sudah ada di notebook asli.
- GNN-ODE dan LSTM dijalankan **live** — mendukung skenario contoh + horizon custom dari
  slider.
- Jika notebook di-rerun dengan data/model baru, cukup jalankan ulang cell Flow 5 dan
  timpa folder `gradio_artifacts/` — tidak perlu ubah `app.py`.
