# PT01 - Computer Vision Application: Object Detection & Tracking

Proyek tugas/praktikum Computer Vision untuk deteksi objek dan multi-object tracking (MOT) menggunakan **YOLOv8** dan **ByteTrack**.

---

## 🚀 Setup & Menjalankan Proyek dengan `uv`

Proyek ini dikelola menggunakan paket manajer modern [uv](https://docs.astral.sh/uv/).

### 1. Sinkronisasi Environment & Dependensi
Pastikan `uv` sudah terpasang. Di direktori proyek, jalankan:
```bash
uv sync
```
Perintah ini akan secara otomatis membuat virtual environment `.venv` dengan Python 3.11 dan menginstal semua library (`ultralytics`, `lapx`, `opencv-python`, `pandas`, `torch`, `torchvision`, `ipykernel`).

### 2. Menjalankan Tracking via CLI ([main.py](main.py))
Jalankan skrip inferensi default:
```bash
uv run python main.py
```

Atau tentukan argumen kustom:
```bash
uv run python main.py --source videos/Normal_Videos_606_x264.mp4 --weights weights/yolov8n.pt --conf 0.4 --tracker bytetrack.yaml
```

Hasil pelacakan akan disimpan di folder `runs/track/hasil_deteksi/`.

### 3. Menjalankan via Jupyter Notebook ([LP03_model_weights.ipynb](LP03_model_weights.ipynb))
Kernel untuk Jupyter telah didaftarkan dengan nama **`Python (PT01 CV uv)`**.
Buka file notebook di VS Code atau browser, lalu pilih kernel **`Python (PT01 CV uv)`** atau arahkan interpreter ke `.venv/Scripts/python.exe`.

---

## 📁 Struktur Direktori
```text
PT01_ComputerVisionApplication/
├── pyproject.toml              # Definisi proyek & dependensi uv
├── uv.lock                     # Lockfile dependensi yang dapat direproduksi
├── .python-version             # Versi Python proyek (3.11)
├── main.py                     # Skrip CLI untuk deteksi dan tracking
├── LP03_model_weights.ipynb    # Jupyter Notebook eksperimen
├── weights/
│   └── yolov8n.pt              # Bobot model YOLOv8 nano
├── videos/
│   └── Normal_Videos_606_x264.mp4 # Video input sampel
└── runs/
    └── track/hasil_deteksi/    # Output video hasil tracking
```
