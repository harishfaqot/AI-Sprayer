# Menyambungkan Alat ke WiFi

Alat ini bisa disambungkan ke WiFi **tanpa laptop dan tanpa kabel**, mirip cara menyambungkan lampu pintar atau CCTV WiFi. Jika alat tidak punya WiFi yang dikenal, alat membuat **WiFi sendiri**. Dari HP, pengguna memilih WiFi lahan/rumah dan mengisi password.

## Cara memakai (untuk pengguna)

1. **Nyalakan alat** dan tunggu **sekitar 2 menit**.
2. Di HP, buka pengaturan WiFi. Cari WiFi bernama **`AI-Sprayer-Setup-xxxx`** (xxxx = 4 huruf/angka berbeda untuk tiap alat).
3. Sambungkan, password: **`sprayer1234`**.
4. Halaman **"Prototype Sprayer Weeder · Pengaturan WiFi"** terbuka sendiri. Jika tidak, buka browser dan ketik **`http://10.42.0.1`**.
5. Pilih nama WiFi dari daftar, isi **password WiFi**, lalu tekan **Sambungkan**.
   - WiFi tidak ada di daftar atau WiFi tersembunyi? Pilih **"WiFi lain / WiFi tersembunyi"** dan ketik namanya.
6. WiFi buatan alat akan **mati sekitar 1 menit** (HP otomatis terputus, itu normal).
7. **Berhasil:** WiFi `AI-Sprayer-Setup` hilang. Sambungkan HP ke WiFi yang sama, lalu buka **`http://ai-sprayer.local`** di browser.
8. **Gagal** (misalnya password salah): dalam ±1 menit `AI-Sprayer-Setup-xxxx` muncul lagi. Sambungkan lagi, dan halamannya menampilkan **alasan gagalnya**. Coba lagi.

## Hal yang dikerjakan alat sendiri

- **Hilang WiFi** (WiFi lahan mati, sinyal jauh): setelah sekitar 30 detik alat membuat WiFi sendiri supaya masih bisa dipakai.
- **Setelah dinyalakan:** alat memberi waktu ±45 detik untuk menyambung ke WiFi yang pernah dipakai sebelum membuat WiFi sendiri.
- **Mencoba lagi sendiri:** jika tidak ada yang memakai WiFi buatan alat, tiap **3 menit** alat mencoba menyambung kembali ke WiFi yang pernah dipakai (misalnya WiFi HP yang tadinya dimatikan). Jadi tidak perlu dinyalakan ulang.
- **Password salah tidak merusak apa-apa:** pengaturan WiFi lama dikembalikan seperti semula.
- **Kabel LAN:** jika alat tersambung lewat kabel LAN, alat tidak membuat WiFi sendiri.

## Tidak ada WiFi sama sekali (pakai hotspot alat saja)

Di lahan yang tidak ada WiFi, dashboard tetap bisa dipakai lewat WiFi buatan alat. **Tidak perlu internet.**

1. Nyalakan alat, tunggu sekitar 2 menit.
2. Sambungkan HP ke **`AI-Sprayer-Setup-xxxx`** (password **`sprayer1234`**).
3. Di halaman yang terbuka, tekan tombol biru **Buka Dashboard**. Dashboard tampil, lalu tekan **▶ MULAI**.
4. Jika HP bertanya *"WiFi ini tidak ada internet, tetap terhubung?"*, pilih **Ya / Tetap terhubung**. Pilihan ini perlu, kalau tidak HP bisa pindah ke data seluler dan dashboard tidak terbuka.
5. Lain kali bisa langsung buka **`http://10.42.0.1:5000`** di browser, selama HP masih tersambung ke WiFi alat.

Hal yang perlu diketahui:
- WiFi buatan alat **tetap menyala selama masih ada HP yang tersambung**. Jika tidak ada HP tersambung selama 3 menit, WiFi itu mati sekitar 40 detik untuk mencari WiFi lama, lalu menyala lagi. HP perlu disambungkan kembali jika terputus.
- Tombol **Buka Dashboard** menghilangkan peringatan "perlu login" di HP tersebut selama 24 jam atau sampai alat dihidupkan ulang.
- Untuk pindah ke WiFi lahan atau rumah, buka `http://10.42.0.1` dan pilih WiFi seperti biasa.

## Jika ada masalah

| Masalah | Yang bisa dicoba |
|---|---|
| WiFi `AI-Sprayer-Setup` tidak muncul | Tunggu sampai **3 menit** setelah alat menyala. Pastikan tidak ada WiFi lama yang sedang menyala dan terjangkau, karena kalau ada alat akan menyambung ke sana. HP yang sedang menjadi hotspot tidak bisa melihat daftar WiFi, jadi cek dari HP/laptop lain. |
| Halaman tidak terbuka otomatis | Buka browser, ketik `http://10.42.0.1`. |
| `ai-sprayer.local` tidak terbuka | Pastikan HP/laptop di WiFi yang **sama** dengan alat, dan program `main_web.py` sedang berjalan. HP Android yang sangat lama kadang tidak mendukung alamat `.local`; coba dari laptop atau iPhone. |
| Sudah mencoba semuanya | Matikan alat 10 detik, nyalakan lagi, tunggu 3 menit. |

---

## Untuk pengembang (teknis, boleh dilewati)

Cara kerja: `wifi_portal.py` berjalan sebagai layanan *systemd* (`wifi-portal.service`) memakai NetworkManager. Jika WiFi putus, layanan membuat *access point* di `wlan0` (jaringan `10.42.0.1`), menyajikan halaman portal di port 80, dan DNS membelokkan semua nama ke alat sehingga HP menampilkan halaman login otomatis.

| File | Fungsi |
|---|---|
| `wifi_portal.py` | Program utama (hanya memakai pustaka bawaan Python). |
| `wifi-portal.service` | Layanan yang menjalankan `wifi_portal.py` saat boot, dengan *watchdog* agar otomatis dihidupkan ulang jika macet. |
| `sprayer-port80.service` | Meneruskan port 80 ke port 5000 (aplikasi sprayer) agar alamat tanpa nomor port bisa dipakai. Alamat hotspot `10.42.0.1` dikecualikan agar portal tetap jalan. |
| `install.sh` | Pemasangan di Orange Pi: `sudo sh install.sh`. |
| `test_wifi_portal.py` | Uji otomatis untuk bagian logika (`python -m unittest test_wifi_portal`). |

Pengaturan (nama WiFi, password, waktu tunggu) ada di `/etc/wifi-portal.json` di alat. Contoh: `{"ap_password": "passwordbaru123", "retry_every": 120, "eth_counts": false}`. Nilai bawaan ada di bagian `DEFAULTS` di `wifi_portal.py`.

Catatan kerja: log layanan ada di `/var/lib/wifi-portal/wifi-portal.log` (tetap ada setelah alat di-restart). Memaksa WiFi buatan menyala untuk uji: `sudo touch /run/wifi-portal-force-ap`.
