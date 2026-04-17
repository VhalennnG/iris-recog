## Issue #4: Iris Boundary Segmentation (Inner & Outer)

**Status:** `To Do` | **Label:** `Core-Engine`, `Math-Heavy`, `Computer-Vision`

### 📝 Description

Mengimplementasikan algoritma untuk mendeteksi dua batas kritis: **Pupillary Boundary** (batas pupil-iris) dan **Limbic Boundary** (batas iris-sklera). Kita akan menggunakan pendekatan _Circular Hough Transform_ (CHT) atau _Daugman’s Integro-differential Operator_ untuk akurasi maksimal.

### 🛠️ Technical Logic

Operator Integro-differential Daugman bekerja dengan mencari nilai maksimum dari turunan parsial integral garis lingkaran:
$$\max_{(r, x_0, y_0)} \left| G_{\sigma}(r) * \frac{\partial}{\partial r} \oint_{r, x_0, y_0} \frac{I(x, y)}{2\pi r} ds \right|$$
Di mana $I(x, y)$ adalah citra asli dan $G_{\sigma}(r)$ adalah fungsi Gaussian untuk penghalusan.

### 🛠️ Tasks

- [ ] Implementasi fungsi `find_pupil()` menggunakan CHT atau Daugman's operator pada citra yang sudah di-_enhance_.
- [ ] Implementasi fungsi `find_iris_outer_boundary()` dengan radius pencarian yang lebih besar.
- [ ] Tambahkan logika deteksi _occlusion_ sederhana untuk mengabaikan area kelopak mata atas/bawah.
- [ ] Optimasi performa menggunakan NumPy _vectorization_ agar pemrosesan tetap di bawah ambang batas waktu.

### ✅ Acceptance Criteria

- [ ] Algoritma berhasil mendeteksi pusat $(x, y)$ dan radius $r$ untuk kedua lingkaran.
- [ ] Toleransi kesalahan deteksi pusat lingkaran $< 2\%$ dari total diameter iris.

---

## Issue #5: Iris Normalization (Daugman’s Rubber Sheet Model)

**Status:** `To Do` | **Label:** `Core-Engine`, `Math-Heavy`

### 📝 Description

Mentransformasi area iris yang berbentuk cincin (_annular region_) menjadi blok persegi panjang dengan ukuran tetap (**$64 \times 512$**). Hal ini dilakukan untuk mengompensasi variasi ukuran pupil akibat kontraksi/dilatasi (cahaya) dan jarak kamera.

### 🛠️ Technical Logic

Memetakan setiap titik pada koordinat Kartesian $(x, y)$ di area iris ke koordinat Polar $(r, \theta)$:
$$x(r, \theta) = (1 - r)x_p(\theta) + rx_i(\theta)$$
$$y(r, \theta) = (1 - r)y_p(\theta) + ry_i(\theta)$$
Di mana $(x_p, y_p)$ dan $(x_i, y_i)$ adalah titik pada batas pupil dan iris.

### 🛠️ Tasks

- [ ] Implementasi modul `core/normalization.py` dengan fungsi `unwrap_iris()`.
- [ ] Gunakan **Bilinear Interpolation** saat melakukan _mapping_ koordinat untuk menghindari _aliasing_.
- [ ] Pastikan output selalu memiliki dimensi yang konsisten: **$64 \times 512$** px.

### ✅ Acceptance Criteria

- [ ] Citra _unwrapped_ terlihat linear dan tekstur iris terdistribusi merata secara horizontal.
- [ ] Fungsi mengembalikan dua objek: `normalized_iris` dan `noise_mask` (untuk menandai area bulu mata/pantulan cahaya).

---

## Issue #6: Visual Validation & Debugging Suite

**Status:** `To Do` | **Label:** `Utility`, `Testing`

### 📝 Description

Membuat alat bantu visualisasi untuk memverifikasi apakah pipeline segmentasi dan normalisasi berjalan dengan benar sebelum masuk ke tahap encoding (Phase 3).

### 🛠️ Tasks

- [ ] Buat script `tools/visualize_pipeline.py` yang menampilkan:
  1. Citra asli dengan _overlay_ lingkaran pupil & iris (warna berbeda).
  2. Hasil _unwrapped iris_ dalam bentuk persegi panjang.
- [ ] Implementasi unit test untuk memastikan output normalisasi tidak mengandung nilai `NaN` atau `Inf`.
- [ ] Jalankan `uv run ruff check` untuk memastikan standarisasi kode pada modul baru.

### ✅ Acceptance Criteria

- [ ] Pengembang dapat melihat hasil segmentasi secara visual untuk keperluan _debugging_.
- [ ] Modul lolos _code coverage_ minimal 80% sesuai standar yang ditetapkan.

---
