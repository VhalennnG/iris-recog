import numpy as np
from typing import Tuple


def _create_log_gabor_filter(
    n_points: int, center_freq: float, sigma_on_f: float
) -> np.ndarray:
    """
    Membuat 1D Log-Gabor filter di frequency domain.

    Filter bekerja sebagai bandpass yang menangkap informasi tekstur pada
    frekuensi tertentu. DC component di-nolkan untuk menghilangkan
    pengaruh pencahayaan global.

    Args:
        n_points: Jumlah titik (lebar baris citra).
        center_freq: Frekuensi pusat filter (0 < f0 < 0.5).
        sigma_on_f: Rasio bandwidth — mengontrol lebar bandpass.

    Returns:
        Array 1D (half-spectrum, panjang n_points//2 + 1) berisi
        magnitude filter di frequency domain.
    """
    n_half = n_points // 2 + 1
    freq = np.linspace(0, 0.5, n_half)

    # Hindari log(0) pada komponen DC
    freq[0] = 1.0

    log_gabor = np.exp(
        -((np.log(freq / center_freq)) ** 2)
        / (2 * (np.log(sigma_on_f)) ** 2)
    )

    # Set DC ke 0 — kita tidak butuh informasi brightness rata-rata
    log_gabor[0] = 0.0

    return log_gabor


def extract_features(
    normalized_iris: np.ndarray,
    noise_mask: np.ndarray,
    n_scales: int = 4,
    min_wave_length: float = 18.0,
    sigma_on_f: float = 0.5,
    mult: float = 1.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ekstraksi fitur iris menggunakan bank of 1D Log-Gabor filters.

    Pipeline:
    1. Untuk setiap skala, buat Log-Gabor filter di frequency domain.
    2. Aplikasikan filter ke setiap baris citra normalisasi via FFT.
    3. Kuantisasi fase (real ≥ 0 → 1, imag ≥ 0 → 1) → 2 bit per titik.
    4. Gabungkan bit dari semua skala menjadi IrisCode.

    Args:
        normalized_iris: Citra iris yang sudah di-unwrap (height × width),
                         biasanya 64 × 512.
        noise_mask: Mask dari tahap normalisasi (1 = valid, 0 = noise).
        n_scales: Jumlah skala frekuensi (default: 4).
        min_wave_length: Panjang gelombang minimum (skala terkecil / frekuensi
                         tertinggi).
        sigma_on_f: Bandwidth ratio untuk Log-Gabor.
        mult: Faktor pengali antar skala (geometric scaling).

    Returns:
        Tuple (iris_code, mask_code):
        - iris_code: np.ndarray boolean, shape (n_scales * 2, height, width).
          Untuk setiap skala s:
            iris_code[2*s]   = bit dari komponen Real (Re ≥ 0)
            iris_code[2*s+1] = bit dari komponen Imag (Im ≥ 0)
        - mask_code: np.ndarray boolean, shape (n_scales * 2, height, width).
          Noise mask yang di-propagasi ke semua channel bit.
    """
    height, width = normalized_iris.shape[:2]

    # Pastikan input float untuk FFT
    image = normalized_iris.astype(np.float64)

    # Alokasi output
    iris_code = np.zeros((n_scales * 2, height, width), dtype=bool)
    mask_code = np.zeros((n_scales * 2, height, width), dtype=bool)

    # Konversi noise_mask ke boolean (1 = valid → True)
    base_mask = noise_mask.astype(bool)

    for s in range(n_scales):
        # Hitung frekuensi pusat untuk skala ini
        wave_length = min_wave_length * (mult**s)
        center_freq = 1.0 / wave_length

        # Buat filter di frequency domain
        log_gabor = _create_log_gabor_filter(width, center_freq, sigma_on_f)

        # Aplikasikan filter ke setiap baris secara vektorisasi
        # rfft pada axis=1 berarti FFT per baris
        image_fft = np.fft.rfft(image, axis=1)

        # Kalikan di frequency domain (konvolusi = perkalian di freq)
        filtered_fft = image_fft * log_gabor[np.newaxis, :]

        # Kembali ke spatial domain — hasilnya bilangan kompleks
        filtered = np.fft.irfft(filtered_fft, n=width, axis=1)

        # Phase Quantization: 2 bit per titik
        # Bit 1: Re(z) ≥ 0
        # Bit 2: Im(z) ≥ 0
        # Karena irfft menghasilkan real, kita perlu Hilbert-like approach:
        # Gunakan full complex response dari filter
        complex_response = np.fft.irfft(
            filtered_fft * 1.0, n=width, axis=1
        )

        # Untuk mendapatkan komponen imajiner, gunakan teknik:
        # hilbert = ifft(fft * 2 * step) dimana step = [0,1,1,...,1,0.5]
        # Tapi lebih simpel: gunakan ifft biasa (bukan irfft) dari full spectrum
        # Bangun full spectrum dari half spectrum
        full_spectrum = np.zeros((height, width), dtype=complex)
        full_spectrum[:, :len(log_gabor)] = image_fft * log_gabor[np.newaxis, :]

        # Buat analytic signal: nolkan frekuensi negatif, gandakan positif
        analytic_spectrum = np.zeros_like(full_spectrum)
        analytic_spectrum[:, 0] = full_spectrum[:, 0]  # DC tetap
        if width % 2 == 0:
            analytic_spectrum[:, width // 2] = full_spectrum[:, width // 2]
            analytic_spectrum[:, 1 : width // 2] = (
                2 * full_spectrum[:, 1 : width // 2]
            )
        else:
            analytic_spectrum[:, 1 : (width + 1) // 2] = (
                2 * full_spectrum[:, 1 : (width + 1) // 2]
            )

        # IFFT untuk mendapatkan analytic signal (complex)
        analytic_signal = np.fft.ifft(analytic_spectrum, axis=1)

        # Phase quantization
        iris_code[2 * s] = np.real(analytic_signal) >= 0
        iris_code[2 * s + 1] = np.imag(analytic_signal) >= 0

        # Propagasi mask ke kedua channel bit
        mask_code[2 * s] = base_mask
        mask_code[2 * s + 1] = base_mask

    return iris_code, mask_code
