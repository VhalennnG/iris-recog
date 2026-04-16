## Issue #1: Project Scaffolding & Modern Tooling Setup

**Status:** `Backlog` | **Priority:** `High` | **Label:** `DevOps`, `Quality-Control`

### 📝 Description

[cite_start]Menyiapkan fondasi repositori menggunakan **uv** sebagai _package manager_ untuk memastikan _environment_ yang deterministik dan cepat, serta **ruff** untuk _linting_ dan _formatting_ agar kode tetap bersih dan konsisten sesuai standar industri[cite: 57, 74].

### 🛠️ Tasks

- [cite_start][ ] **Initialize Project:** Jalankan `uv init` untuk membuat struktur dasar dan `uv venv` untuk isolasi virtual environment[cite: 58, 75].
- [ ] **Dependency Management:** Tambahkan pustaka inti menggunakan `uv add`:
  - [cite_start]`opencv-python`, `numpy`, `scipy` (Core CV & Math)[cite: 76].
  - [cite_start]`fastapi`, `uvicorn[standard]` (API wrapper)[cite: 77].
  - [cite_start]`ruff`, `pytest`, `pre-commit` (Developer tools)[cite: 77].
- [cite_start][ ] **Linting & Formatting:** Konfigurasi `ruff` dalam `pyproject.toml` sebagai pengganti Black, Isort, dan Flake8[cite: 78].
- [cite_start][ ] **Folder Structure:** Inisialisasi struktur modular berikut[cite: 61, 80]:
  - `/core`: Algoritma utama (segmentasi, encoding).
  - `/api`: Endpoint FastAPI.
  - `/data`: Script pemuatan dataset.
  - `/tests`: Unit testing.
- [cite_start][ ] **Git Hooks:** Setup `.pre-commit-config.yaml` agar `ruff` berjalan otomatis sebelum setiap _commit_[cite: 79].

### ✅ Acceptance Criteria

- [cite_start][ ] Proyek dapat di-instansiasi secara instan hanya dengan perintah `uv sync`[cite: 64, 81].
- [cite_start][ ] File `uv.lock` tersedia untuk menjamin reproduksibilitas antar pengembang[cite: 60, 82].
- [ ] Perintah `uv run ruff check .` tidak menghasilkan peringatan atau error.

---

## Issue #2: High-Performance Dataset Pipeline

**Status:** `Backlog` | **Priority:** `High` | **Label:** `Data-Engineering`, `Computer-Vision`

### 📝 Description

Membangun modul pemuatan data yang efisien untuk dataset **CASIA-Iris-Interval** atau **UBIRIS.v2**. [cite_start]Fokus pada kecepatan pembacaan citra mentah dan metadata terkait (ID Subjek, Sisi Mata)[cite: 29, 30, 83, 84].

### 🛠️ Tasks

- [cite_start][ ] Buat _class_ `IrisDataset` di `core/dataset.py` yang mendukung _lazy loading_ untuk menghemat memori[cite: 85].
- [cite_start][ ] Implementasi mekanisme pemisahan data (_train-test split_) berdasarkan **Subject ID** untuk menghindari _data leakage_ (jangan sampai gambar mata yang sama ada di kedua set)[cite: 34, 86].
- [cite_start][ ] Tambahkan unit test dasar menggunakan `pytest` untuk memverifikasi integritas jumlah gambar dan format citra[cite: 65, 87].

### ✅ Acceptance Criteria

- [cite_start][ ] Script berhasil memetakan ribuan citra ke dalam objek NumPy dalam hitungan detik menggunakan `uv run`[cite: 35, 88].
- [cite_start][ ] Mendukung pemuatan citra dalam format `.jpg`, `.bmp`, dan `.tiff` secara konsisten[cite: 32].

---

## Issue #3: Image Enhancement & Noise Reduction Module

**Status:** `Backlog` | **Priority:** `High` | **Label:** `Core-Engine`, `Math-Heavy`

### 📝 Description

Implementasi pipeline awal pemrosesan citra untuk memperbaiki kualitas iris sebelum segmentasi. [cite_start]Fokus utama adalah reduksi _noise_ dengan tetap menjaga ketajaman tepi (_edge-preserving_) yang krusial untuk deteksi pupil[cite: 38, 39, 89].

### 🛠️ Technical Logic

[cite_start]Penggunaan **Bilateral Filter** sangat direkomendasikan karena kemampuannya meredam _noise_ tanpa mengaburkan batas lingkaran iris[cite: 41, 90]. Rumus dasarnya adalah:

$$I_{filtered}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\| ) g_s(\|x_i - x\|)$$

### 🛠️ Tasks

- [cite_start][ ] Implementasi fungsi `enhance_iris_visibility()` di modul `core/preprocessing.py`[cite: 43, 92].
- [cite_start][ ] Integrasi **CLAHE** (_Contrast Limited Adaptive Histogram Equalization_) untuk menonjolkan tekstur halus pada iris yang memiliki pencahayaan buruk[cite: 42, 45, 93].
- [cite_start][ ] Gunakan `pytest` untuk memastikan fungsi pengolahan citra selalu mengembalikan _array_ dengan dimensi yang benar[cite: 94].

### ✅ Acceptance Criteria

- [cite_start][ ] Hasil visual menunjukkan kontras tekstur iris yang jauh lebih tajam dibandingkan citra mentah (_raw_)[cite: 46, 95].
- [cite_start][ ] Modul memiliki dokumentasi _docstrings_ lengkap dengan tipe data (_Type Hinting_) standar Python 3.11+[cite: 18, 96].

---

> [!TIP]
> **Iris's Pro-Tip:** Menggunakan `uv` memungkinkan Anda menjalankan eksperimen preprocessing langsung melalui terminal tanpa aktivasi manual. [cite_start]Cukup gunakan `uv run python script.py` untuk iterasi cepat saat mencari nilai $\sigma$ yang paling pas untuk filter[cite: 97, 98].
