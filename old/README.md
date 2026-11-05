# Arsip Percobaan Lama

Folder ini berisi **semua percobaan sebelum program akhir** (`main_web.py` di folder utama). Isinya disimpan sebagai catatan perjalanan penelitian dan bahan laporan. **Tidak dipakai lagi oleh alat.**

Cerita lengkap tiap tahap ada di [README utama](../README.md#4-perjalanan-penelitian-urut-waktu).

## Cara menjalankan program lama

Program lama mencari video dan model dengan alamat relatif. Jadi **masuk dulu ke folder `old`** sebelum menjalankan:

```
cd old
python main_yolo.py
```

> Beberapa program lama mencari file yang sudah tidak ada atau berpindah tempat (misalnya `grass.mp4` di folder yang sama). Itu wajar: program dibiarkan apa adanya sebagai arsip. Jika ingin menjalankan, sesuaikan nama file di bagian atas program.

## Isi arsip, urut waktu

### Tahap 1 — Mei 2026: Mengenali rumput dari warna
| File | Fungsi |
|---|---|
| `adaptive_hsv.py` | Mengenali rumput dari warna hijau, otomatis menyesuaikan terang-gelap. Tanpa tampilan jendela. |
| `hsv.py` | Versi paling sederhana, untuk mencoba-coba batas warna dengan slider. |
| `hsv_gui.py` | Tampilan lengkap (PyQt5) dengan panel slider dan angka langsung. |
| `main_color.py` | Tampilan 4 panel (asli, topeng, hasil) yang memutar semua video di folder `video/`. |
| `demo.mp4` | Video demo tahap ini. |

### Tahap 2 — 3 Juni 2026: Mencoba AI siap pakai
| File | Fungsi |
|---|---|
| `yolo_test.py` | Mencoba YOLO (pengenal objek) memakai `people.mp4`. |
| `depth.py` | Memperkirakan kedalaman dengan Depth Anything V2. |
| `midas.py`, `midas2.py` | Memperkirakan kedalaman dengan MiDaS (model ringan). |
| `grass_segment.py` | Memisahkan rumput dengan model SegFormer siap pakai. |
| `models/`, `midas_small.onnx` | File model untuk MiDaS. |

### Tahap 3 — 7 Juni 2026: Melatih AI sendiri
| File / folder | Fungsi |
|---|---|
| `main_yolo.py` | Menjalankan model YOLO buatan sendiri: menandai rumput, menentukan pompa dan bukaan nozzle (tampilan Tkinter). |
| `clean_unlabeled.py` | Membuang gambar yang belum ditandai dari dataset. |
| `split_dataset.py` | Membagi dataset menjadi bagian untuk belajar dan untuk menguji. |
| `command.txt` | Perintah yang dipakai untuk melatih model. |
| `Datasets/Grass_Segmentation/` | Data latihan pertama (rumput, 185 gambar). |
| `Datasets/images/` | Gambar mentah sebelum ditandai. |
| `runs/segment/train` … `train-6` | Hasil pelatihan model-model pertama (nilai ketepatan 50–60%). |
| `video/grass4.mp4` … `grass7.mp4`, `video/DEMO_7JUN2026.mp4` | Video uji dan demo. |

### Tahap 4 — 6 Juli 2026: Kamera dari atas
| File / folder | Fungsi |
|---|---|
| `main_yolo_vertical.py` | Versi untuk kamera dari atas: menentukan arah nozzle dari posisi rumput terhadap titik tengah gambar. |
| `Datasets/Vertical_Grass_Segmentation/` | Data latihan sudut atas (222 gambar). |
| `Datasets/CVAT_Vertical_Grass_Segmentation.zip` | Hasil ekspor dari alat menandai gambar (CVAT). |
| `Datasets/images_vertical/` | Gambar mentah sudut atas. |
| `runs/segment/train-7`, `train-8`, `train-8.zip` | Hasil pelatihan sudut atas (`train-8` ≈ 83%). |
| `runs/detect/` | Percobaan mengenali dengan kotak (bukan mewarnai area). |
| `yolo26n.pt`, `yolo26n-seg.pt`, `yolo26n-sem.pt`, `yoloe-26n-seg.pt` | Model dasar YOLO sebagai titik awal belajar. |
| `video/vgrass1.mp4` … `vgrass3.MP4`, `demo_6Jul2026.mp4` | Video sudut atas dan demo. |

### Tahap 5 — Oktober 2026
| File | Fungsi |
|---|---|
| `extract_frames_10.py` | Mengambil 1 gambar setiap 10 bingkai dari `real1-3.mp4` (di folder utama `video/`) untuk ditandai. Alat bantu pembuatan data terakhir. |

### Tahap 6 — Oktober 2026
| Folder | Isi |
|---|---|
| `logs/` | 9 catatan percobaan PID pertama (file CSV, bisa dibuka di Excel). Dibuat dengan pengaturan awal, jadi bentuk kolomnya berbeda dari versi sekarang. Catatan baru dibuat otomatis di folder `logs/` utama. |

### Lain-lain
| File | Fungsi |
|---|---|
| `people.mp4` | Video orang, hanya untuk mencoba YOLO. |
