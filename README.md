# Prototype Sprayer Weeder 🌿 — Penyemprot Gulma Otomatis

Penelitian disertasi: membuat **alat penyemprot yang bekerja sendiri**. Kamera melihat lahan, kecerdasan buatan (AI) mengenali gulma, lalu alat **mengarahkan nozzle ke gulma** dan **mengatur banyak-sedikitnya semprotan**. Tujuannya: herbisida hanya disemprotkan di tempat yang ada gulma, sehingga lebih hemat dan lebih ramah lingkungan.

> Dokumen ini ditulis untuk pembaca non-teknis (misalnya mahasiswa pertanian). Istilah teknis dijelaskan di bagian [Kamus Istilah](#kamus-istilah).

---

## 1. Cara kerja alat (versi sederhana)

```
 Kamera ──► AI mengenali gulma ──► Hitung: di mana & seberapa banyak ──► Gerakkan alat
 (video)    (menandai area gulma)   (kiri/kanan? sedikit/banyak?)         (arah nozzle + bukaan)
```

1. **Kamera** merekam lahan.
2. **AI** menandai bagian gambar yang berupa gulma (warna merah di layar).
3. Program menghitung **posisi gulma** (di kiri, tengah, atau kanan) dan **seberapa luas** gulma itu.
4. Dua **servo** (motor kecil yang bisa berhenti di sudut tertentu) bergerak:
   - **Servo sudut** → memutar nozzle ke kiri/kanan, mengarah ke gulma.
   - **Servo nozzle** → membuka lubang semprot: gulma sedikit → bukaan kecil, gulma banyak → bukaan besar.
5. Agar gerakan servo **halus dan tidak menyentak**, dipakai pengatur yang disebut **PID** (lihat Kamus Istilah).

Semua ini bisa dipantau lewat **halaman web (dashboard)** dari HP atau laptop.

---

## 2. Isi folder

| Folder / file | Isinya (bahasa sehari-hari) |
|---|---|
| `main_web.py` | **Program utama.** Menjalankan AI, dashboard web, dan kontrol servo. Inilah yang dipakai di alat. |
| `dashboard.html` | Tampilan halaman web (dashboard). Berisi teks dan tata letak; bisa diubah tanpa menyentuh program utama. |
| `pid_control.py` | Pengatur gerak servo (PID), uji coba respons (*step test*), dan pencatat data ke file. Dipakai oleh `main_web.py`. |
| `runs/segment/train-9/` | **Model AI terbaru** (hasil belajar AI dari video lapangan asli). File yang dipakai: `weights/best.pt` (laptop) dan `weights/best_416.onnx` (Orange Pi, versi yang lebih cepat dan dibuat dari `best.pt`; tidak disimpan di GitHub). |
| `Datasets/real_segmentation/` | **Data latihan terbaru**: gambar lahan asli + tanda gulma yang dibuat manual. |
| `video/` | Video lapangan terbaru: `real1.mp4`, `real2.mp4`, `real3.mp4` (ukuran asli 2560×1440), salinan ringan `real1_720p.mp4` … `real3_720p.mp4` (1280×720, dipakai di Orange Pi), dan `frames_all/` (potongan gambar dari video itu). |
| `wifi_portal/` | Agar alat bisa disambungkan ke WiFi tanpa laptop. Baca [`wifi_portal/README.md`](wifi_portal/README.md). |
| `deploy/` | File pemasangan di Orange Pi agar `main_web.py` menyala otomatis saat alat dihidupkan. |
| `STL/` | File desain 3D untuk bagian mekanik (roda gigi). |
| `old/` | **Arsip percobaan lama** (semua program, data, dan model sebelum yang sekarang). Baca [`old/README.md`](old/README.md). |
| `requirements.txt` | Daftar program tambahan yang harus dipasang di komputer. |
| `logs/` | Dibuat otomatis saat program jalan: rekaman data gerakan servo (file CSV, bisa dibuka di Excel). |

> ⚠️ **Video asli `real1.mp4`, `real2.mp4`, `real3.mp4` tidak disimpan di GitHub** karena ukurannya lebih dari 100 MB per file (batas GitHub). Simpan salinannya di tempat lain (flash disk / Google Drive). Salinan 720p (`*_720p.mp4`, hanya 6–8 MB) ikut disimpan di GitHub sehingga program tetap bisa dicoba tanpa video asli.

---

## 3. Cara menjalankan

### Di laptop / PC
1. Pasang **Python** (versi 3.10 atau lebih baru).
2. Buka terminal di folder ini, lalu pasang kebutuhan program:
   ```
   pip install -r requirements.txt
   ```
3. Pastikan file `video/real1.mp4` ada.
4. Jalankan:
   ```
   python main_web.py
   ```
5. Buka browser ke **http://localhost:5000**, lalu tekan tombol **▶ MULAI**.

Pengaturan (video yang diputar, model AI, dll.) ada di bagian atas file `main_web.py`, di bawah tulisan `CONFIG`.

### Di Orange Pi 4 Pro (komputer kecil di alat)
- **Otomatis menyala saat alat dihidupkan.** Tunggu sekitar 1–2 menit, lalu buka dari HP/laptop di WiFi yang sama: **http://ai-sprayer.local** (tanpa perlu tahu alamat IP, tanpa nomor port). Halaman dashboard langsung tampil; tekan **▶ MULAI** untuk memulai deteksi.
- **Tidak ada WiFi?** Alat membuat WiFi sendiri dan dashboard tetap bisa dibuka. Lihat [`wifi_portal/README.md`](wifi_portal/README.md), bagian "Tidak ada WiFi sama sekali".
- Menyambungkan alat ke WiFi baru: lihat [`wifi_portal/README.md`](wifi_portal/README.md).
- Program di Orange Pi berada di folder `/home/orangepi/sprayer-weeder`. Setelah mengubah program di laptop, salin ulang `main_web.py`, `pid_control.py`, dan `dashboard.html` ke folder itu, lalu jalankan `sudo systemctl restart sprayer-web`.
- (Teknis) Model cepat `best_416.onnx` dibuat dengan perintah `yolo export model=runs/segment/train-9/weights/best.pt format=onnx imgsz=416 opset=12 simplify=False`, lalu file hasilnya diganti namanya menjadi `best_416.onnx`. Pengaturan model, ukuran gambar, dan video di Orange Pi ada di `deploy/sprayer-web.service` (baris `Environment=`).

### Memakai kamera (bukan video)

> ⚠️ Bagian ini **belum diuji dengan kamera sungguhan**: saat dokumen ini ditulis belum ada kamera yang dicolok ke Orange Pi. Jalur programnya sudah disiapkan; hasilnya perlu dicek saat kamera tersedia.

1. **Pakai webcam USB** (hampir semua webcam standar "UVC" bisa). Lebih baik yang mendukung format **MJPG 1280×720, 30 fps**. Colokkan ke port USB Orange Pi, sebaiknya port USB 3.0 (biasanya berwarna biru).
2. **Cek kamera terdeteksi.** Di Orange Pi (lewat SSH), jalankan `ls /dev/video*`. Harus muncul `/dev/video0` (webcam sering membuat dua, `video0` dan `video1`; yang dipakai yang pertama). Untuk melihat format yang didukung: `v4l2-ctl --list-formats-ext -d /dev/video0`.
3. **Di dashboard**, kolom **SUMBER VIDEO**: pilih **0** (tertulis "Kamera"), lalu tekan **▶ MULAI**.
   - Muncul pesan *"Tidak bisa membuka sumber video: 0"*? Kamera tidak terbaca. Coba angka `1` atau `2`, cabut dan colok lagi kameranya, atau periksa dengan langkah 2.
4. **Agar kamera menjadi sumber bawaan saat alat dihidupkan:** di `deploy/sprayer-web.service` ganti baris `Environment=SPRAYER_SOURCE=video/real1_720p.mp4` menjadi `Environment=SPRAYER_SOURCE=0`, salin file itu ke Orange Pi, lalu jalankan `sudo sh install_web.sh`. Deteksi tetap baru mulai setelah tombol MULAI ditekan.
5. **Resolusi kamera** bawaannya 1280×720 pada 30 fps. Bisa diubah dengan baris `Environment=SPRAYER_CAM_W=…`, `SPRAYER_CAM_H=…`, `SPRAYER_CAM_FPS=…` di file layanan yang sama. Resolusi lebih besar membuat Orange Pi lebih berat dan lebih panas.

**Pemasangan kamera sangat memengaruhi hasil.** Model AI belajar dari video `real1`–`real3` yang direkam dari atas, menghadap ke bawah, dengan roda alat terlihat di kiri dan kanan. Pasang kamera dengan sudut dan tinggi yang mirip. Jika sudutnya berbeda, hasil deteksi bisa turun dan posisi **area semprot** (persegi panjang di video) perlu disesuaikan: pengaturannya ada di bagian `CONFIG` file `main_web.py` (`ROI_Y_CENTER`, `ROI_H`, `NOZZLE_HEIGHT`).

**Kamera jenis lain** (misalnya kamera dengan kabel pita MIPI/CSI) memakai cara pemasangan berbeda dan belum dibahas atau diuji di sini. Jika kameramu jenis itu, beri tahu merek/tipenya.

### Membaca tampilan dashboard
Halaman web punya tiga tab di bagian atas:

**1. Pantau** (halaman utama, cukup ini untuk pemakaian sehari-hari)
- **Tombol ▶ MULAI / ■ BERHENTI** ada di pojok kanan atas. Di sebelahnya ada angka **FPS** (gambar per detik) dan **suhu** alat. Suhu berwarna kuning/merah berarti alat terlalu panas (lihat bagian kecepatan di bawah).
- **Video** menandai gulma dengan area berwarna dan nama jenisnya: **oranye = Rumput Teki**, **ungu = Babandotan**, **biru = Rumput Belulang**. Persegi panjang hijau/kuning = **area semprot**; segitiga biru-muda = **semprotan**; tanda **+** merah = titik bidik. Klik "Arti warna dan garis di video" untuk daftar lengkap.
- **SUMBER VIDEO**: pilih file video dari daftar atau ketik `0` untuk kamera. Di Orange Pi pakai salinan 720p, misalnya `video/real2_720p.mp4`. Video asli 2560×1440 terlalu berat untuk Orange Pi.
- **Kondisi sekarang**: tiga kotak besar: ada/tidaknya **gulma**, pompa **NYALA/MATI**, dan **bukaan nozzle** (%, lengkap dengan targetnya).
- **Arah nozzle**: gambar arah semprotan, serta sudut sekarang dan sudut target.
- **Pengukuran**: luas gulma di seluruh gambar dan di area semprot.
- **Penggunaan sistem**: seberapa sibuk CPU (%), pemakaian memori (RAM), suhu (°C), dan waktu yang dipakai AI untuk memproses satu gambar (ms). Makin kecil waktu itu, makin lancar deteksinya.

**2. Kontrol PID (lanjutan)**: untuk peneliti. Grafik garis merah putus-putus (target) dan biru (kenyataan); makin dekat kedua garis, makin bagus. Di sini nilai Kp/Ki/Kd bisa diubah. Tombol **UJI STEP** memberi target tetap untuk mengukur seberapa bagus pengaturannya, hasilnya muncul sebagai tabel.

**3. Catatan**: riwayat kejadian (mulai, berhenti, perubahan pengaturan).

Di HP, semua kotak tersusun ke bawah. Tampilan halaman selalu terang (tema terang), apa pun pengaturan HP atau laptop.

### Kecepatan deteksi (FPS) di Orange Pi

Di laptop deteksi bisa mencapai 30 FPS (gambar per detik). Di Orange Pi awalnya hanya 1–2 FPS. Penyebab dan perbaikannya, semuanya sudah diukur langsung di alat:

| Penyebab | Perbaikan | Hasil |
|---|---|---|
| AI dijalankan di CPU dengan PyTorch. | Model diubah ke format **ONNX** (lebih cepat di CPU jenis ini) dan gambar masuk AI diperkecil dari 640 ke **416** piksel. | Waktu AI per gambar turun dari ±590 ms menjadi ±150–190 ms. |
| Video asli 2560×1440 terlalu berat dibuka bersamaan dengan AI. Saat keduanya jalan, AI melambat sekitar 4 kali (165 ms → 646 ms). | Memakai salinan **720p**. Dengan video 720p AI hanya melambat sedikit (161 ms → 193 ms). | AI tidak lagi terganggu. |
| Membaca video dan menjalankan AI dilakukan bergantian di satu "pekerja". | Membaca video dipisah ke pekerja sendiri, sehingga AI selalu memakai gambar terbaru. | Video tetap berjalan sesuai waktu nyata. |
| Suhu CPU naik sampai ±90 °C. | Perlu perangkat keras (lihat di bawah). | |

**Hasil sekarang:** ±5 FPS saat alat masih dingin, turun menjadi ±3 FPS setelah sekitar setengah menit karena panas.

**Pilihan ukuran gambar masuk AI.** Makin kecil, makin cepat, tetapi tepi area gulma makin kurang pas. Ketepatan dihitung pada 59 gambar uji (ingat: gambar uji ini juga dipakai untuk memilih model terbaik, jadi angkanya cenderung sedikit lebih bagus dari kenyataan):

| Ukuran | Waktu AI per gambar (ONNX) | Perkiraan FPS | Gulma ditemukan dengan benar (mAP50) | Tepi area pas (mAP50-95) |
|---|---|---|---|---|
| 640 | ±330 ms | ±3 | 95,1% | 89,9% |
| 480 | tidak diukur | – | 95,6% | 87,6% |
| **416 (dipakai)** | **±150 ms** | **±6,7** | **96,4%** | **86,5%** |
| 320 | ±95 ms | ±10 | 95,6% | 82,4% |

Artinya: kemampuan menemukan gulma hampir tidak berubah di semua ukuran; yang turun hanya kepasan tepi area. Untuk menyemprot, ukuran 320 mungkin masih cukup dan memberi ±10 FPS. Ini perlu diuji di lahan. FPS di dashboard sedikit lebih rendah dari tabel karena ada pekerjaan lain (menghitung luas gulma, menggambar, mengirim video).

**Suhu.** Orange Pi mulai memperlambat dirinya pada 70 °C dan lebih kuat lagi di 90 °C. Saat deteksi berjalan suhunya mencapai ±90 °C, dan kipas pendingin tidak terdeteksi di sistem. **Pasang heatsink dan kipas**; ini cara termurah untuk mempertahankan FPS.

**NPU (chip khusus AI, 3 TOPS).** Orange Pi 4 Pro punya NPU dan drivernya sudah terdeteksi di sistem, tetapi **belum dipakai**. NPU berpotensi memberi lompatan terbesar (puluhan FPS), namun butuh pekerjaan tambahan:
- Model harus diubah ke format khusus NPU memakai alat dari Allwinner di PC Linux (dengan Docker), lalu dikecilkan ke angka 8-bit. Ketepatannya harus diuji ulang.
- Contoh resminya memakai bahasa C++, sehingga perlu disambungkan ke `main_web.py`.
- Sebagai gambaran: dokumentasi Radxa (papan lain dengan chip Allwinner A733 yang sejenis) melaporkan model YOLOv8s-seg di NPU: 38 ms untuk AI saja, ±77 ms per gambar dengan persiapan dan pasca-proses (±13 FPS). Model yang lebih kecil akan lebih cepat, tetapi belum diukur di alat ini.
- Model kita (YOLO26) belum tentu bisa diubah. Kemungkinan perlu melatih ulang model YOLOv8n-seg dengan data yang sama.

---

## 4. Perjalanan penelitian (urut waktu)

Tanggal di bawah adalah tanggal pengiriman ke GitHub (*commit*), sehingga bisa sedikit berbeda dari tanggal pengerjaan sebenarnya.

### Tahap 1 — 11 Mei 2026: Mengenali rumput dari warna
- **Apa yang dicoba:** mengenali rumput hanya dari **warna hijaunya** (metode HSV), lalu menghitung berapa persen gambar yang tertutup rumput.
- **Hasil:** program dengan beberapa tampilan (Tkinter, PyQt5, OpenCV) dan penyesuaian otomatis terhadap terang-gelap cahaya. Video uji: `grass`, `grass2`, `grass3`.
- **Keterbatasan metode ini:** hanya melihat warna, jadi semua yang hijau dianggap sama. Hasilnya mudah berubah mengikuti cahaya.
- **Arsip:** `old/adaptive_hsv.py`, `hsv.py`, `hsv_gui.py`, `main_color.py`, `old/demo.mp4`.

### Tahap 2 — 3 Juni 2026: Mencoba AI yang sudah jadi
- **Apa yang dicoba:** memakai model AI siap pakai tanpa melatih sendiri:
  - **YOLO** (pengenal objek umum) — dites dengan video orang (`people.mp4`) sebagai latihan memakai YOLO.
  - **MiDaS dan Depth Anything** — memperkirakan **jarak/kedalaman** dari satu kamera.
  - **SegFormer** — AI siap pakai yang memisahkan gambar menjadi bagian-bagian, termasuk rumput.
- **Hasil:** percobaan pengetahuan. Tidak ada yang dipakai di program akhir, tetapi pengalaman ini membawa ke tahap berikutnya (melatih AI sendiri).
- **Arsip:** `old/yolo_test.py`, `old/depth.py`, `old/midas.py`, `old/midas2.py`, `old/grass_segment.py`.

### Tahap 3 — 7 Juni 2026: Melatih AI sendiri dan dashboard web pertama
- **Apa yang dikerjakan:**
  - Mengambil gambar rumput (212 gambar mentah), menandai area rumput secara manual (**labeling**) → data latihan `Grass_Segmentation` (185 gambar: 148 untuk belajar, 37 untuk menguji).
  - Melatih model **YOLO segmentasi** beberapa kali (`train` sampai `train-6`).
  - Membuat `main_yolo.py`: AI menandai rumput, memperkirakan tinggi rumput, lalu menentukan **pompa NYALA/MATI** dan **bukaan nozzle 0–100%**.
  - Membuat **`main_web.py` pertama** — dashboard di browser. Ini cikal bakal program utama sekarang.
  - Alat bantu data: `clean_unlabeled.py` (membuang gambar yang belum ditandai), `split_dataset.py` (membagi data belajar/uji).
  - Video baru `grass4`–`grass7`, video demo `DEMO_7JUN2026.mp4`, dan desain 3D roda gigi (`STL/gear72.3mf`).
- **Hasil pelatihan** (nilai ketepatan, lihat tabel di bawah): sekitar **50–60%**.
- **Arsip:** `old/main_yolo.py`, `old/clean_unlabeled.py`, `old/split_dataset.py`, `old/command.txt` (perintah pelatihan), `old/runs/`.

### Tahap 4 — 6 Juli 2026: Kamera dari atas
- **Apa yang dikerjakan:**
  - Pindah ke **sudut pandang dari atas** (kamera menghadap ke bawah, lebih cocok untuk menyemprot).
  - Data baru `Vertical_Grass_Segmentation` (222 gambar: 178 belajar, 44 uji). Pelabelan dibuat dengan alat CVAT.
  - Melatih `train-7` dan `train-8`; **`train-8` memberi nilai sekitar 83%**.
  - `main_yolo_vertical.py`: menghitung **posisi rumput terhadap titik tengah gambar** untuk menentukan arah nozzle, dan luas rumput untuk bukaan nozzle.
  - Memakai model dasar terbaru (YOLO26) dan video demo `demo_6Jul2026.mp4`.
- **Catatan hati-hati:** pada data ini ukuran lain dari nilai ketepatan bahkan mendekati 100%. Itu kemungkinan karena datanya sedikit dan gambarnya mirip satu sama lain, jadi belum tentu setinggi itu di lahan nyata.
- **Arsip:** `old/main_yolo_vertical.py`, `old/Datasets/`, `old/runs/segment/train-8/`.

### Tahap 5 — 11 Oktober 2026: Data lapangan asli
- **Apa yang dikerjakan:**
  - Merekam **video lahan asli**: `real1`, `real2`, `real3`.
  - Mengambil 1 gambar setiap 10 bingkai video → **307 gambar** (`video/frames_all/`).
  - Menandai gulma secara manual, kali ini **3 jenis gulma** (data latihan `Datasets/real_segmentation`: 248 belajar, 59 uji):

    | Nama lokal | Nama ilmiah (umum dikenal) |
    |---|---|
    | Rumput Teki | *Cyperus rotundus* |
    | Babandotan | *Ageratum conyzoides* |
    | Rumput Belulang | *Eleusine indica* |
  - Melatih **`train-9`** → **nilai sekitar 93%**, tertinggi sejauh ini.
  - `main_web.py` diubah memakai `train-9` dan video `real1.mp4`.
- **Inilah model dan video yang dipakai sekarang.**

### Tahap 6 — 12 Oktober 2026: Gerak servo yang halus (kontrol PID)
- **Apa yang dikerjakan:** membuat `pid_control.py` dan menyambungkannya ke dashboard.
  - Dua servo (sudut dan bukaan nozzle) **disimulasikan di komputer** — belum memakai servo asli.
  - Grafik target-vs-kenyataan langsung di dashboard, bisa mengubah nilai Kp/Ki/Kd, dan tombol **UJI STEP**.
  - Setiap jalan, data gerakan dicatat ke file CSV di folder `logs/` (untuk grafik di laporan). Catatan 9 percobaan pertama ada di `old/logs/`.
- **Hasil uji dengan pengaturan sekarang** (simulasi):
  - Servo sudut naik ke target dalam **kurang lebih 0,3 detik** dan hanya **melewati target maksimal ±2%**.
  - Servo nozzle serupa: sekitar 0,25–0,3 detik, kelebihan maksimal ±2%.
  - Selisih akhir dari target kurang dari 0,5 (derajat atau persen).
  - Artinya gerakan cepat, tidak bergoyang. Ini masih **simulasi**; angka dengan servo asli perlu diukur lagi.

### Tahap 7 — 5 November 2026: Alat di Orange Pi 4 Pro
- **Apa yang dikerjakan:**
  - Memasang sistem di **Orange Pi 4 Pro** (komputer kecil di alat, Ubuntu 22.04).
  - **WiFi portal** (`wifi_portal/`): jika alat tidak terhubung WiFi, alat membuat WiFi sendiri bernama `AI-Sprayer-Setup-xxxx`. Pengguna menyambungkan HP, lalu memilih WiFi dan mengisi password lewat halaman yang terbuka otomatis (seperti WiFi Manager di ESP32).
  - Alat bisa dibuka lewat nama **`ai-sprayer.local`** tanpa perlu tahu alamat IP, dan **tanpa mengetik nomor port** (`http://ai-sprayer.local`). Halaman ini baru tampil jika `main_web.py` sedang berjalan di Orange Pi.
  - Program `main_web.py` **otomatis menyala saat alat dihidupkan** (layanan `sprayer-web`), sudah diuji dengan menghidupkan ulang alat.
  - Jika tidak ada WiFi, dashboard tetap bisa dibuka lewat WiFi buatan alat (tombol **Buka Dashboard** di halaman setup).
- **Status:** sudah diuji di alat. File-file ini belum dikirim ke GitHub saat dokumen ini ditulis.

---

### Ringkasan hasil model AI

"**Nilai ketepatan**" di sini artinya: seberapa pas area gulma yang ditandai AI dibandingkan tanda buatan manusia (100% = sama persis). Dalam istilah teknis ini disebut *mAP50-95 (mask)*. Yang dicantumkan adalah nilai terbaik selama pelatihan.

| Model | Data latihan | Sudut kamera | Nilai ketepatan | Catatan |
|---|---|---|---|---|
| `train-2` | Grass_Segmentation | samping/depan | 50% | Berhenti di putaran ke-19 dari 100 |
| `train-3` | Grass_Segmentation | samping/depan | 57% | |
| `train-4` | Grass_Segmentation | samping/depan | 55% | |
| `train-5` | Grass_Segmentation | samping/depan | 60% | Terbaik di kelompok ini |
| `train-6` | Grass_Segmentation | samping/depan | 55% | |
| `train-8` | Vertical_Grass_Segmentation | atas | 83% | Data sedikit, hati-hati |
| **`train-9`** | **real_segmentation** | **lapangan asli** | **93%** | **Dipakai sekarang** |

`train` dan `train-7` tidak punya hasil pelatihan yang lengkap (tidak selesai).

---

## 5. Yang belum selesai

- [ ] **Servo asli belum dipasang.** Gerakan servo masih simulasi; PID perlu diuji ulang dengan servo dan nozzle sungguhan.
- [ ] **Kamera asli di Orange Pi belum dipasang dan diuji.** Saat ini program memutar video `real1_720p.mp4`. Panduannya ada di bagian "Memakai kamera".
- [ ] **Pasang pendingin (heatsink + kipas) di Orange Pi.** Tanpa itu suhu mencapai ±90 °C dan FPS turun.
- [ ] **Coba NPU** supaya deteksi jauh lebih cepat (lihat bagian kecepatan di atas). Perlu keputusan: melatih ulang model YOLOv8n-seg agar cocok dengan alat konversi NPU.
- [ ] Menentukan ukuran gambar masuk AI (416 atau 320) berdasarkan uji di lahan.
- [ ] **Deteksi belum mulai otomatis.** Dashboard sudah tampil sendiri saat alat dihidupkan, tetapi deteksi baru berjalan setelah tombol **▶ MULAI** ditekan.
- [ ] **Program di Orange Pi disalin manual** dari laptop (folder `/home/orangepi/sprayer-weeder`), belum ditarik lewat GitHub. Salinan lama di `/home/orangepi/AI-Sprayer` tidak dipakai lagi.
- [ ] Uji **penyemprotan nyata** di lahan dan perhitungan penghematan herbisida.
- [ ] Menambah data latihan dari lahan dengan kondisi berbeda (cahaya, jenis tanah, tanaman utama) agar model tidak hanya bagus di 3 video ini.

---

## Kamus Istilah

| Istilah | Artinya |
|---|---|
| **AI / model** | Program komputer yang sudah "belajar" mengenali sesuatu dari banyak contoh gambar. |
| **Segmentasi** | AI mewarnai tepat bagian gambar yang dikenali (seperti mewarnai bentuk gulma), bukan hanya menggambar kotak. |
| **Dataset / data latihan** | Kumpulan gambar yang dipakai AI untuk belajar, beserta tanda yang dibuat manual. |
| **Labeling** | Menandai manual bagian gambar mana yang gulma, sebagai "kunci jawaban" untuk AI. |
| **Melatih (training)** | Proses AI belajar dari dataset. Satu putaran belajar disebut *epoch*. |
| **Servo** | Motor kecil yang bisa berhenti tepat di sudut yang diminta (dipakai untuk memutar dan membuka nozzle). |
| **Nozzle** | Ujung penyemprot tempat cairan keluar. |
| **PID** | Pengatur otomatis supaya gerakan servo ke target **cepat tapi halus**: jauh dari target → bergerak cepat, mendekat → melambat dan berhenti pas. Ada tiga pengatur: **Kp** (dorongan utama), **Ki** (menghilangkan sisa selisih kecil), **Kd** (rem agar tidak kebablasan). |
| **Target / kenyataan** | Target = sudut/bukaan yang *diminta* kamera. Kenyataan = posisi servo *sekarang*. |
| **Step test** | Uji dengan memberi target yang berubah mendadak, untuk melihat seberapa cepat dan stabil servo mengikutinya. |
| **Overshoot** | Servo melewati target sedikit sebelum berhenti (makin kecil makin baik). |
| **Dashboard** | Halaman web untuk memantau dan mengontrol alat. |
| **Orange Pi** | Komputer sebesar kartu kredit yang dipasang di alat. |
| **Hotspot / portal WiFi** | WiFi buatan alat sendiri, dipakai hanya untuk memilih WiFi rumah/lahan. |
| **`.local` (ai-sprayer.local)** | Nama alat di jaringan WiFi, supaya tidak perlu menghafal angka alamat IP. |
| **CSV** | File tabel sederhana; bisa dibuka di Excel. |
| **FPS** | Jumlah gambar yang diproses per detik; makin besar makin lancar. |

---

## Lisensi

MIT
