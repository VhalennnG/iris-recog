## Issue #7: Feature Extraction with 2D Log-Gabor Filters

**Status:** `To Do` | **Priority:** `High` | **Label:** `Core-Engine`, `Math-Heavy`

### 📝 Description

Mengimplementasikan ekstraksi fitur tekstur dari citra iris yang sudah di-normalisasi ($64 \times 512$). Kita menggunakan **2D Log-Gabor Filter** karena kemampuannya menangkap informasi fase tanpa dipengaruhi oleh variasi intensitas cahaya (pencahayaan global).

### 🛠️ Technical Logic

1.  **Filtering:** Konvolusi citra normalisasi dengan filter Log-Gabor pada berbagai skala dan orientasi.
2.  **Phase Quantization:** Hasil filter (bilangan kompleks) dipetakan ke dalam 4 kuadran pada bidang kompleks. Setiap titik menghasilkan 2 bit informasi.
3.  **IrisCode Generation:** Menggabungkan bit-bit tersebut menjadi vektor biner 2048-bit (atau sesuai konfigurasi).

### 🛠️ Tasks

- [ ] Implementasi fungsi `extract_features()` di `core/encoder.py`.
- [ ] Buat generator filter Log-Gabor menggunakan rumus:
      $$G(f) = \exp\left(\frac{-(\log(f/f_0))^2}{2(\log(\sigma/f_0))^2}\right)$$
- [ ] Implementasi kuantisasi fase untuk menghasilkan **IrisCode** dan **Noise Mask** (area yang harus diabaikan saat pencocokan).
- [ ] Optimasi dengan NumPy agar ekstraksi fitur berjalan di bawah 100ms.

### ✅ Acceptance Criteria

- [ ] Output berupa array biner (IrisCode) dan array biner pendamping (Mask).
- [ ] Modul melewati pengujian konsistensi: input yang sama harus selalu menghasilkan IrisCode yang identik.

---

## Issue #8: Matching Logic & Normalized Hamming Distance

**Status:** `To Do` | **Priority:** `High` | **Label:** `Core-Engine`, `Math-Heavy`

### 📝 Description

Membangun algoritma untuk membandingkan dua IrisCode. Karena adanya kemungkinan gangguan (noise) seperti bulu mata atau pantulan cahaya, kita menggunakan **Normalized Hamming Distance** (HD) yang mempertimbangkan bit-masking.

### 🛠️ Technical Logic

Pencocokan dilakukan dengan operasi XOR antara dua kode, namun hanya pada area di mana kedua mask bernilai valid (0 untuk data bersih, 1 untuk noise):
$$HD = \frac{\| (CodeA \oplus CodeB) \cap (MaskA \cup MaskB)^c \|}{\| (MaskA \cup MaskB)^c \|}$$

### 🛠️ Tasks

- [ ] Implementasi fungsi `calculate_hamming_distance(code1, code2, mask1, mask2)`.
- [ ] **Rotation Compensation:** Implementasi pergeseran bit (_bit shifting_) secara horizontal pada IrisCode untuk menangani kemiringan kepala (rotasi ringan) saat pengambilan gambar.
- [ ] Gunakan operasi _bitwise_ NumPy untuk kecepatan maksimal.

### ✅ Acceptance Criteria

- [ ] Perhitungan HD antara dua kode identik menghasilkan nilai $0.0$.
- [ ] Perhitungan HD antara dua kode dari subjek berbeda menghasilkan nilai mendekati $0.5$ (acak).

---

## Issue #9: Accuracy Benchmarking & Threshold Calibration

**Status:** `To Do` | **Priority:** `High` | **Label:** `Data-Science`, `Evaluation`

### 📝 Description

Melakukan pengujian massal menggunakan dataset CASIA/UBIRIS untuk menentukan _threshold_ keputusan yang optimal. Kita perlu menyeimbangkan antara keamanan dan kenyamanan pengguna.

### 🛠️ Tasks

- [ ] Buat script `tools/benchmark_accuracy.py`.
- [ ] Hitung distribusi **Intra-class** (orang yang sama) dan **Inter-class** (orang berbeda).
- [ ] Kalkulasi Metrik:
  - **FAR (False Acceptance Rate):** Orang asing dianggap sebagai pemilik sah.
  - **FRR (False Rejection Rate):** Pemilik sah ditolak oleh sistem.
- [ ] Plot kurva ROC (_Receiver Operating Characteristic_) dan tentukan **EER (Equal Error Rate)**.

### ✅ Acceptance Criteria

- [ ] Menghasilkan nilai _threshold_ optimal (misal: $HD < 0.32$) untuk digunakan pada Phase 4.
- [ ] Laporan akurasi terdokumentasi dengan visualisasi distribusi HD.

---

### 💡 Iris's Pro-Tip:

Saat melakukan _bit shifting_ untuk menangani rotasi, jangan hanya geser satu kali. Biasanya kita melakukan pergeseran sebanyak $\pm 8$ atau $\pm 16$ kolom dan mengambil nilai **Hamming Distance terendah** sebagai hasil akhir. Ini adalah cara paling efektif untuk membuat sistem kamu tetap akurat meskipun pengguna memiringkan kepalanya sedikit.
