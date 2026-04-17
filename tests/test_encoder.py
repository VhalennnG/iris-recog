import numpy as np
import pytest

from core.encoder import _create_log_gabor_filter, extract_features


class TestLogGaborFilter:
    """Unit tests untuk generator Log-Gabor filter."""

    def test_output_shape(self):
        """Filter harus memiliki panjang n_points//2 + 1 (half-spectrum)."""
        n_points = 512
        f = _create_log_gabor_filter(n_points, center_freq=0.05, sigma_on_f=0.5)
        assert f.shape == (n_points // 2 + 1,)

    def test_dc_component_is_zero(self):
        """Komponen DC (index 0) harus 0 — menghilangkan bias pencahayaan."""
        f = _create_log_gabor_filter(512, center_freq=0.05, sigma_on_f=0.5)
        assert f[0] == 0.0

    def test_values_non_negative(self):
        """Semua nilai filter harus >= 0 (magnitude di freq domain)."""
        f = _create_log_gabor_filter(512, center_freq=0.1, sigma_on_f=0.5)
        assert np.all(f >= 0)

    def test_peak_near_center_freq(self):
        """Puncak filter harus berada di sekitar frekuensi pusat."""
        n_points = 512
        center_freq = 0.1
        f = _create_log_gabor_filter(n_points, center_freq, sigma_on_f=0.5)

        freq_axis = np.linspace(0, 0.5, n_points // 2 + 1)
        peak_idx = np.argmax(f[1:]) + 1  # Skip DC
        peak_freq = freq_axis[peak_idx]

        # Puncak harus dekat dengan center_freq (toleransi karena diskretisasi)
        assert abs(peak_freq - center_freq) < 0.02


class TestExtractFeatures:
    """Unit tests untuk pipeline ekstraksi fitur."""

    @pytest.fixture
    def synthetic_iris(self):
        """Membuat citra iris sintetis dengan tekstur periodik."""
        height, width = 64, 512
        x = np.linspace(0, 8 * np.pi, width)
        # Tekstur bergelombang di setiap baris
        texture = np.sin(x)[np.newaxis, :] * np.ones((height, 1))
        # Tambahkan sedikit noise realistis
        rng = np.random.default_rng(42)
        texture += rng.normal(0, 0.1, (height, width))
        # Normalisasi ke range 0-255
        texture = ((texture - texture.min()) / (texture.max() - texture.min()) * 255)
        return texture.astype(np.uint8)

    @pytest.fixture
    def valid_mask(self, synthetic_iris):
        """Mask tanpa noise (semua area valid)."""
        return np.ones(synthetic_iris.shape, dtype=np.uint8)

    def test_output_types(self, synthetic_iris, valid_mask):
        """Output harus berupa boolean arrays."""
        code, mask = extract_features(synthetic_iris, valid_mask)
        assert code.dtype == bool
        assert mask.dtype == bool

    def test_output_shape(self, synthetic_iris, valid_mask):
        """Shape output harus (n_scales * 2, height, width)."""
        n_scales = 4
        code, mask = extract_features(synthetic_iris, valid_mask, n_scales=n_scales)
        h, w = synthetic_iris.shape
        expected = (n_scales * 2, h, w)
        assert code.shape == expected
        assert mask.shape == expected

    def test_consistency(self, synthetic_iris, valid_mask):
        """Input yang sama harus selalu menghasilkan IrisCode identik."""
        code1, _ = extract_features(synthetic_iris, valid_mask)
        code2, _ = extract_features(synthetic_iris, valid_mask)
        np.testing.assert_array_equal(code1, code2)

    def test_different_inputs_differ(self, valid_mask):
        """Dua input yang berbeda harus menghasilkan IrisCode yang berbeda."""
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (64, 512), dtype=np.uint8)
        img2 = rng.integers(0, 256, (64, 512), dtype=np.uint8)

        code1, _ = extract_features(img1, valid_mask)
        code2, _ = extract_features(img2, valid_mask)

        # Kode harus berbeda (tidak identik)
        assert not np.array_equal(code1, code2)

    def test_mask_propagation(self, synthetic_iris):
        """Noise mask harus ter-propagasi ke semua channel bit."""
        mask = np.ones(synthetic_iris.shape, dtype=np.uint8)
        # Tandia baris 0 sebagai noise
        mask[0, :] = 0

        _, mask_code = extract_features(synthetic_iris, mask, n_scales=2)

        # Semua channel di baris 0 harus False (noise)
        for ch in range(4):  # 2 scales * 2 bits
            assert not np.any(mask_code[ch, 0, :])

        # Baris lain harus True (valid)
        for ch in range(4):
            assert np.all(mask_code[ch, 1:, :])

    def test_custom_scales(self, synthetic_iris, valid_mask):
        """Jumlah skala harus bisa dikonfigurasi."""
        for n_scales in [1, 2, 6]:
            code, mask = extract_features(
                synthetic_iris, valid_mask, n_scales=n_scales
            )
            assert code.shape[0] == n_scales * 2
