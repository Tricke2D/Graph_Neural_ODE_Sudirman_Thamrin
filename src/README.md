## Struktur

```
src/
├── data/
│   ├── collect\_road\_network.py     # OSMnx: ambil road network dari OSM
│   └── generate\_synthetic\_speed.py # generator data kecepatan sintetis
├── models/
│   ├── ode\_func.py                 # f\_theta(h,t) = GCN(h), diintegrasikan odeint
│   ├── gnn\_ode.py                  # Encoder -> ODEFunc -> Decoder
│   └── lstm\_forecast.py            # baseline neural non-graph
└── utils/
    └── metrics.py                  # MAE, RMSE, normalisasi, breakdown peak/off-peak
```

## Konsistensi dengan app.py

Class di `models/` sengaja ditulis **identik** dengan definisi di `app.py`
(nama layer, urutan operasi, default parameter) supaya kalau suatu saat
`app.py` mau direfactor untuk `import` dari `src/`, tidak ada risiko
`state\_dict` gagal dimuat karena arsitektur berbeda.

