## Implement High-Precision Iris Recognition Engine (Biometric Core)

**Status:** `Backlog` | **Priority:** `High` | **Label:** `Core-Engine`, `Computer-Vision`, `Math-Heavy`

### 📝 Description

Membangun _core engine_ untuk pengenalan biometrik berbasis iris mata. Sistem ini akan bertanggung jawab untuk mengekstraksi identitas unik dari citra mata manusia dan mengubahnya menjadi representasi biner (**IrisCode**) yang efisien untuk disimpan dan dicocokkan.

### 🛠️ Technical Stack & Architecture

- **Language:** Python 3.11+
- **Core CV:** OpenCV (Image Processing), NumPy & SciPy (Linear Algebra & Matrix Ops).
- **Feature Extraction:** Log-Gabor Filters (untuk ekstraksi tekstur).
- **Database/Persistence:** PostgreSQL dengan ekstensi `pgvector` (untuk indexing dan pencarian vektor biner).
- **Deployment Ready:** FastAPI (sebagai wrapper microservice).
- **Dataset (Development):** [CASIA-Iris-Interval](http://www.cbsr.ia.ac.cn/english/IrisDatabase.asp) atau UBIRIS.v2.

---

### 🧬 Model & Algorithm Logic

Sistem ini tidak menggunakan deteksi objek generik, melainkan _pipeline_ pemrosesan sinyal yang spesifik:

1.  **Segmentation:**
    - **Model:** _Circular Hough Transform_ (CHT) atau _Daugman’s Integro-differential Operator_.
    - **Goal:** Mendeteksi dua lingkaran konsentris: batas pupil-iris dan batas iris-sklera.

2.  **Normalization:**
    - **Model:** _Daugman’s Rubber Sheet Model_.
    - **Goal:** Memetakan area iris dari koordinat kartesian $(x, y)$ ke koordinat polar $(r, \theta)$ untuk menangani variasi ukuran pupil akibat cahaya.

3.  **Feature Extraction:**
    - **Model:** _2D Log-Gabor Filters_.
    - **Goal:** Mendekomposisi tekstur iris menjadi informasi fase. Hasil akhirnya adalah **IrisCode** (vektor biner 2048-bit).

4.  **Matching:**
    - **Logic:** _Normalized Hamming Distance_.
    - **Metric:** Menghitung perbedaan bit antara dua IrisCode dengan mengabaikan area _noise_ (seperti bulu mata) menggunakan bit-masking.

---

### 🗺️ Development Roadmap

#### Phase 1: Environment & Pre-processing (Week 1)

- [ ] Setup repositori dengan struktur modular.
- [ ] Implementasi _Image Enhancement_ (Grayscale conversion, Bilateral Filtering untuk reduksi noise tanpa merusak tepi).
- [ ] Pipeline loading dataset CASIA/UBIRIS.

#### Phase 2: Segmentation & Normalization (Week 2)

- [ ] Implementasi algoritma segmentasi untuk isolasi iris.
- [ ] Implementasi _Rubber Sheet Model_ untuk transformasi polar.
- [ ] Validasi visual hasil _unwrapped iris_ (harus berbentuk persegi panjang $64 \times 512$).

#### Phase 3: Biometric Encoding & Matching (Week 3)

- [ ] Implementasi Log-Gabor filter untuk ekstraksi fitur.
- [ ] Pembuatan modul kalkulasi _Hamming Distance_.
- [ ] Pengujian akurasi: Menentukan _threshold_ optimal untuk _False Acceptance Rate_ (FAR) dan _False Rejection Rate_ (FRR).

#### Phase 4: Persistence & API Integration (Week 4)

- [ ] Integrasi PostgreSQL + `pgvector` untuk penyimpanan IrisCode.
- [ ] Pembuatan API Endpoint `/register` dan `/verify` menggunakan FastAPI.
- [ ] Dockerization untuk _deployment_ microservice.

---

### ✅ Acceptance Criteria

- [ ] Sistem dapat memproses citra mata mentah hingga menjadi IrisCode dalam < 300ms.
- [ ] Berhasil melakukan identifikasi subjek yang sama dari sudut pandang yang berbeda (invariant terhadap rotasi ringan).
- [ ] Modul terdokumentasi dengan _docstrings_ standar Google/NumPy.
- [ ] _Code coverage_ untuk unit testing minimal 80%.

---
