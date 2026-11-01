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

### 4. Deteksi Objek Hilang ([soal3_lost_objects.py](soal3_lost_objects.py))
```bash
uv run python soal3_lost_objects.py
```

Sebuah ID dianggap hilang jika absen selama `--patience` frame berturut-turut (default 15, atau 0,5 detik). Hasil disimpan di `runs/lost_objects/`:
- `lost_objects_log.csv`: satu baris per kejadian hilang, berisi posisi terakhir, alasan, dan status kembali.
- `lost_objects_per_frame.csv`: ID terlihat dan ID hilang pada tiap frame.
- `lost_objects_output.mp4`: video dengan penanda merah putus-putus di posisi terakhir objek hilang.

---

## 📁 Struktur Direktori
```text
PT01_ComputerVisionApplication/
├── pyproject.toml              # Definisi proyek & dependensi uv
├── uv.lock                     # Lockfile dependensi yang dapat direproduksi
├── .python-version             # Versi Python proyek (3.11)
├── main.py                     # Skrip CLI untuk deteksi dan tracking
├── soal3_lost_objects.py       # Skrip deteksi objek hilang
├── LP03_model_weights.ipynb    # Jupyter Notebook eksperimen
├── weights/
│   └── yolov8n.pt              # Bobot model YOLOv8 nano
├── videos/
│   └── Normal_Videos_606_x264.mp4 # Video input sampel
└── runs/
    ├── track/hasil_deteksi/    # Output video hasil tracking
    └── lost_objects/           # Output log dan video objek hilang
```
