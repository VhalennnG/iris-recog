import numpy as np
from typing import Tuple


def calculate_hamming_distance(
    code1: np.ndarray,
    code2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> float:
    """
    Menghitung Normalized Hamming Distance antara dua IrisCode.

    Formula:
        HD = ||(CodeA ⊕ CodeB) ∩ ¬(MaskA ∪ MaskB)|| / ||¬(MaskA ∪ MaskB)||

    Dimana:
    - ⊕ = XOR (bit yang berbeda)
    - Mask: True = valid (data bersih), False = noise
    - Hanya bit yang valid di KEDUA kode yang dihitung

    Args:
        code1: IrisCode pertama (boolean array).
        code2: IrisCode kedua (boolean array, shape sama dengan code1).
        mask1: Noise mask pertama (boolean array, True = valid).
        mask2: Noise mask kedua (boolean array, True = valid).

    Returns:
        Normalized Hamming Distance (float).
        0.0 = identik, ~0.5 = acak (subjek berbeda), 1.0 = invers sempurna.
    """
    # Area yang valid di kedua kode (AND karena mask True = valid)
    valid_bits = mask1 & mask2

    n_valid = np.count_nonzero(valid_bits)
    if n_valid == 0:
        # Tidak ada bit valid untuk perbandingan
        return 1.0

    # XOR menghasilkan True di posisi bit yang berbeda
    disagreement = (code1 ^ code2) & valid_bits

    return float(np.count_nonzero(disagreement)) / float(n_valid)


def match_iris_codes(
    code1: np.ndarray,
    code2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    max_shift: int = 8,
) -> Tuple[float, int]:
    """
    Membandingkan dua IrisCode dengan rotation compensation.

    Karena pengguna mungkin sedikit memiringkan kepala saat pengambilan
    gambar, iris yang sama bisa ter-rotasi beberapa derajat. Pada IrisCode
    (citra yang sudah di-unwrap), rotasi ini muncul sebagai pergeseran
    horizontal. Fungsi ini mencoba semua kemungkinan shift dalam range
    [-max_shift, +max_shift] dan mengambil HD terendah.

    Args:
        code1: IrisCode pertama, shape (n_channels, height, width).
        code2: IrisCode kedua, shape sama.
        mask1: Mask pertama.
        mask2: Mask kedua.
        max_shift: Jumlah kolom maksimum untuk pergeseran (default: 8).
                   Sesuai rekomendasi, nilai ±8 sudah cukup untuk rotasi
                   kepala ringan. Naikkan ke ±16 untuk variasi lebih besar.

    Returns:
        Tuple (min_hd, best_shift):
        - min_hd: Hamming Distance terkecil yang ditemukan.
        - best_shift: Jumlah kolom shift yang menghasilkan HD terkecil
                      (negatif = geser kiri, positif = geser kanan).
    """
    min_hd = 1.0
    best_shift = 0

    for shift in range(-max_shift, max_shift + 1):
        if shift == 0:
            shifted_code2 = code2
            shifted_mask2 = mask2
        else:
            # np.roll pada axis terakhir (kolom = arah angular)
            shifted_code2 = np.roll(code2, shift, axis=-1)
            shifted_mask2 = np.roll(mask2, shift, axis=-1)

        hd = calculate_hamming_distance(code1, shifted_code2, mask1, shifted_mask2)

        if hd < min_hd:
            min_hd = hd
            best_shift = shift

    return min_hd, best_shift
