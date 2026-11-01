import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# 1. SETUP
# ==========================================
VIDEO_INPUT_PATH = "Normal_Videos_606_x264.mp4" 
MODEL_PATH = "yolov8n.pt"
VIDEO_OUTPUT_PATH = "output_visualisasi_dicky.mp4"
GRAFIK_OUTPUT_PATH = "tren_jumlah_objek.png"

# Inisialisasi Model
model = YOLO(MODEL_PATH)

# Inisialisasi Video Capture
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Setup Video Writer untuk menyimpan hasil overlay
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

# List untuk menyimpan data grafik
frame_numbers = []
object_counts = []
frame_idx = 0

print("Memulai pemrosesan frame...")

# ==========================================
# 2. COUNTING & VISUALISASI PER FRAME
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Jalankan deteksi objek pada frame saat ini
    results = model(frame, verbose=False)
    
    # Hitung jumlah objek yang terdeteksi
    # results[0].boxes berisi kumpulan bounding box dari objek yang terdeteksi
    detections = results[0].boxes
    count = len(detections)
    
    # Simpan data untuk grafik
    frame_numbers.append(frame_idx)
    object_counts.append(count)
    
    # Buat visualisasi: Gambar bounding box bawaan YOLO
    annotated_frame = results[0].plot()
    
    # Buat visualisasi: Tambahkan teks jumlah objek di sudut kiri atas
    text = f"Total Objek: {count}"
    cv2.putText(annotated_frame, text, (40, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    
    # Simpan frame ke video output
    out.write(annotated_frame)
    frame_idx += 1

# Bersihkan resource video
cap.release()
out.release()
print(f"Video visualisasi berhasil disimpan di: {VIDEO_OUTPUT_PATH}")

# ==========================================
# 3. MEMBUAT GRAFIK TREN JUMLAH OBJEK
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, object_counts, color='b', linewidth=1.5)
plt.title("Tren Jumlah Objek per Frame")
plt.xlabel("Nomor Frame")
plt.ylabel("Jumlah Objek Terdeteksi")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Simpan grafik sebagai gambar
plt.savefig(GRAFIK_OUTPUT_PATH, dpi=300)
print(f"Grafik tren berhasil disimpan di: {GRAFIK_OUTPUT_PATH}")
