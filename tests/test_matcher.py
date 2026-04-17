import numpy as np
import pytest

from core.matcher import calculate_hamming_distance, match_iris_codes


class TestCalculateHammingDistance:
    """Unit tests untuk Normalized Hamming Distance."""

    def test_identical_codes_return_zero(self):
        """HD antara dua kode identik harus tepat 0.0."""
        rng = np.random.default_rng(42)
        code = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code, dtype=bool)

        hd = calculate_hamming_distance(code, code, mask, mask)
        assert hd == 0.0

    def test_inverse_codes_return_one(self):
        """HD antara kode dan inversnya harus tepat 1.0."""
        rng = np.random.default_rng(42)
        code = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code, dtype=bool)

        hd = calculate_hamming_distance(code, ~code, mask, mask)
        assert hd == 1.0

    def test_random_codes_near_half(self):
        """HD antara dua kode acak independen harus mendekati 0.5."""
        rng = np.random.default_rng(42)
        code1 = rng.choice([True, False], size=(8, 64, 512))
        code2 = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code1, dtype=bool)

        hd = calculate_hamming_distance(code1, code2, mask, mask)

        # Dengan sampel sebesar ini, HD harus sangat dekat 0.5
        assert 0.45 < hd < 0.55

    def test_mask_excludes_noise(self):
        """Bit yang di-mask harus tidak mempengaruhi hasil HD."""
        code1 = np.array([True, True, False, False])
        code2 = np.array([True, False, True, False])
        # Mask: hanya bit 0 dan 3 yang valid
        mask1 = np.array([True, False, False, True])
        mask2 = np.array([True, False, False, True])

        # Bit valid: index 0 (T vs T = agree), index 3 (F vs F = agree)
        hd = calculate_hamming_distance(code1, code2, mask1, mask2)
        assert hd == 0.0

    def test_partial_mask(self):
        """Verifikasi HD dengan mask parsial."""
        code1 = np.array([True, True, True, False])
        code2 = np.array([False, True, True, False])
        # Semua valid
        mask = np.array([True, True, True, True])

        hd = calculate_hamming_distance(code1, code2, mask, mask)
        # 1 dari 4 bit berbeda
        assert hd == pytest.approx(0.25)

    def test_no_valid_bits_returns_one(self):
        """Jika tidak ada bit valid, kembalikan 1.0 (worst case)."""
        code1 = np.array([True, False])
        code2 = np.array([False, True])
        mask1 = np.array([False, False])
        mask2 = np.array([False, False])

        hd = calculate_hamming_distance(code1, code2, mask1, mask2)
        assert hd == 1.0

    def test_asymmetric_masks(self):
        """HD harus benar saat mask1 dan mask2 berbeda."""
        code1 = np.array([True, False, True])
        code2 = np.array([False, False, True])
        mask1 = np.array([True, True, True])
        mask2 = np.array([True, False, True])

        # Valid bits: index 0 (TvF=disagree), index 2 (TvT=agree)
        # Index 1 invalid karena mask2 False
        hd = calculate_hamming_distance(code1, code2, mask1, mask2)
        assert hd == pytest.approx(0.5)


class TestMatchIrisCodes:
    """Unit tests untuk matching dengan rotation compensation."""

    def test_identical_codes(self):
        """Kode identik harus match sempurna (HD=0, shift=0)."""
        rng = np.random.default_rng(42)
        code = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code, dtype=bool)

        hd, shift = match_iris_codes(code, code, mask, mask)
        assert hd == 0.0
        assert shift == 0

    def test_shifted_code_compensated(self):
        """Kode yang di-shift harus terdeteksi dan HD ≈ 0."""
        rng = np.random.default_rng(42)
        code1 = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code1, dtype=bool)

        # Shift code1 sebanyak 5 kolom ke kanan
        shift_amount = 5
        code2 = np.roll(code1, shift_amount, axis=-1)

        hd, best_shift = match_iris_codes(code1, code2, mask, mask, max_shift=8)

        assert hd == 0.0
        # best_shift harus kompensasi shift_amount (arah berlawanan)
        assert best_shift == -shift_amount

    def test_shift_beyond_range_not_compensated(self):
        """Shift di luar range max_shift tidak boleh ter-kompensasi."""
        rng = np.random.default_rng(42)
        code1 = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code1, dtype=bool)

        # Shift code sebanyak 20 — jauh di luar max_shift=8
        code2 = np.roll(code1, 20, axis=-1)

        hd, _ = match_iris_codes(code1, code2, mask, mask, max_shift=8)

        # HD harusnya tinggi karena shift tidak bisa dikompensasi
        assert hd > 0.1

    def test_random_codes_high_hd(self):
        """Dua kode acak harus menghasilkan HD tinggi (~0.5) setelah matching."""
        rng = np.random.default_rng(42)
        code1 = rng.choice([True, False], size=(8, 64, 512))
        code2 = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code1, dtype=bool)

        hd, _ = match_iris_codes(code1, code2, mask, mask)

        # Dengan rotation compensation, HD minimum bisa sedikit < 0.5
        # tapi tetap harus jauh dari 0
        assert hd > 0.35

    def test_max_shift_zero(self):
        """Dengan max_shift=0, tidak ada compensation — hanya direct match."""
        rng = np.random.default_rng(42)
        code = rng.choice([True, False], size=(8, 64, 512))
        mask = np.ones_like(code, dtype=bool)

        # Shift code
        shifted = np.roll(code, 3, axis=-1)

        hd, shift = match_iris_codes(code, shifted, mask, mask, max_shift=0)

        # Tanpa compensation, HD harus > 0
        assert hd > 0
        assert shift == 0
