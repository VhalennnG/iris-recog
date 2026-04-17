"""
Benchmark Accuracy & Threshold Calibration untuk Iris Recognition.

Menghitung distribusi Intra-class dan Inter-class Hamming Distance,
lalu menentukan threshold optimal berdasarkan Equal Error Rate (EER).

Penggunaan:
    uv run tools/benchmark_accuracy.py <dataset_path> [--max-samples N] [--output-dir DIR]

Contoh:
    uv run tools/benchmark_accuracy.py data/CASIA-Iris-Interval
    uv run tools/benchmark_accuracy.py data/CASIA-Iris-Interval --max-samples 50 --output-dir results/
"""

import argparse
import os
import sys
import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

# Menambahkan root project ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import IrisDataset
from core.encoder import extract_features
from core.matcher import match_iris_codes
from core.normalization import unwrap_iris
from core.preprocessing import enhance_iris_visibility
from core.segmentation import segment_iris


def process_single_image(image: np.ndarray) -> tuple:
    """
    Jalankan pipeline penuh pada satu citra iris.

    Returns:
        Tuple (iris_code, mask_code) atau None jika gagal.
    """
    try:
        # 1. Preprocessing
        enhanced = enhance_iris_visibility(image)

        # 2. Segmentasi
        boundaries = segment_iris(enhanced)

        # 3. Normalisasi
        normalized, noise_mask = unwrap_iris(
            enhanced, boundaries["pupil"], boundaries["iris"]
        )

        # 4. Feature Extraction
        iris_code, mask_code = extract_features(normalized, noise_mask)

        return iris_code, mask_code

    except (ValueError, Exception) as e:
        return None


def run_benchmark(dataset_path: str, max_samples: int = 0, max_inter_pairs: int = 5000):
    """
    Pipeline benchmark utama.

    Args:
        dataset_path: Path ke folder dataset CASIA/UBIRIS.
        max_samples: Batasi jumlah sampel (0 = gunakan semua).
        max_inter_pairs: Batas maksimum pasangan inter-class untuk efisiensi.
    """
    print("=" * 60)
    print("  IRIS RECOGNITION — ACCURACY BENCHMARK")
    print("=" * 60)

    # --- 1. Load Dataset ---
    print(f"\n📁 Loading dataset dari: {dataset_path}")
    dataset = IrisDataset(dataset_path)

    if len(dataset) == 0:
        print("❌ Dataset kosong. Pastikan path benar dan berisi gambar.")
        print("   Format yang didukung: .jpg, .bmp, .tiff")
        print("   Struktur: dataset_path/subject_id/eye_side/image.ext")
        return

    total = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))
    print(f"   Ditemukan {len(dataset)} gambar, memproses {total} sampel.\n")

    # --- 2. Encode Semua Gambar ---
    print("🔄 Menjalankan pipeline (preprocess → segment → normalize → encode)...")

    encoded_data = []  # List of (iris_code, mask_code, subject_id)
    failed = 0
    encode_times = []

    for i in range(total):
        image, subject_id = dataset[i]

        t_start = time.perf_counter()
        result = process_single_image(image)
        t_elapsed = time.perf_counter() - t_start
        encode_times.append(t_elapsed)

        if result is not None:
            iris_code, mask_code = result
            encoded_data.append((iris_code, mask_code, subject_id))
            status = "✓"
        else:
            failed += 1
            status = "✗"

        # Progress bar sederhana
        progress = (i + 1) / total * 100
        print(
            f"\r   [{progress:5.1f}%] {i+1}/{total} "
            f"(berhasil: {len(encoded_data)}, gagal: {failed}) "
            f"— {t_elapsed*1000:.0f}ms",
            end="",
            flush=True,
        )

    print()  # Newline setelah progress

    if len(encoded_data) < 2:
        print("❌ Tidak cukup sampel berhasil diproses untuk benchmarking.")
        return

    avg_time = np.mean(encode_times) * 1000
    print(f"\n⏱️  Rata-rata waktu encoding: {avg_time:.1f}ms per gambar")
    if avg_time < 100:
        print("   ✓ Target < 100ms tercapai!")
    else:
        print(f"   ⚠ Melebihi target 100ms ({avg_time:.0f}ms)")

    # --- 3. Hitung Distribusi HD ---
    print("\n📊 Menghitung Hamming Distance distributions...")

    intra_hd = []  # Pasangan dari subjek yang SAMA
    inter_hd = []  # Pasangan dari subjek BERBEDA

    # Kelompokkan berdasarkan subject_id
    subject_groups = {}
    for code, mask, sid in encoded_data:
        subject_groups.setdefault(sid, []).append((code, mask))

    # --- Intra-class (genuine pairs) ---
    for sid, samples in subject_groups.items():
        if len(samples) < 2:
            continue
        for (c1, m1), (c2, m2) in combinations(samples, 2):
            hd, _ = match_iris_codes(c1, c2, m1, m2)
            intra_hd.append(hd)

    # --- Inter-class (impostor pairs) ---
    subject_ids = list(subject_groups.keys())
    rng = np.random.default_rng(42)

    inter_pairs_generated = 0
    for sid_a, sid_b in combinations(subject_ids, 2):
        if inter_pairs_generated >= max_inter_pairs:
            break

        # Ambil satu sampel acak dari masing-masing subjek
        sa = subject_groups[sid_a][rng.integers(len(subject_groups[sid_a]))]
        sb = subject_groups[sid_b][rng.integers(len(subject_groups[sid_b]))]

        hd, _ = match_iris_codes(sa[0], sb[0], sa[1], sb[1])
        inter_hd.append(hd)
        inter_pairs_generated += 1

    print(f"   Intra-class pairs: {len(intra_hd)}")
    print(f"   Inter-class pairs: {len(inter_hd)}")

    if len(intra_hd) == 0:
        print(
            "❌ Tidak ada intra-class pairs. Pastikan dataset memiliki "
            "lebih dari 1 gambar per subjek."
        )
        return

    intra_hd = np.array(intra_hd)
    inter_hd = np.array(inter_hd)

    # --- 4. Kalkulasi FAR, FRR, EER ---
    print("\n📈 Menghitung FAR, FRR, dan EER...")

    thresholds = np.linspace(0.0, 0.6, 1000)
    far_curve = []
    frr_curve = []

    for t in thresholds:
        # FAR: proporsi inter-class pair yang salah diterima (HD < threshold)
        far = np.mean(inter_hd < t) if len(inter_hd) > 0 else 0.0
        # FRR: proporsi intra-class pair yang salah ditolak (HD >= threshold)
        frr = np.mean(intra_hd >= t) if len(intra_hd) > 0 else 0.0
        far_curve.append(far)
        frr_curve.append(frr)

    far_curve = np.array(far_curve)
    frr_curve = np.array(frr_curve)

    # EER: titik di mana FAR ≈ FRR
    eer_idx = np.argmin(np.abs(far_curve - frr_curve))
    eer = (far_curve[eer_idx] + frr_curve[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    print(f"\n{'='*60}")
    print(f"  📋 HASIL BENCHMARK")
    print(f"{'='*60}")
    print(f"  Sampel berhasil diproses : {len(encoded_data)}/{total}")
    print(f"  Subjek unik              : {len(subject_groups)}")
    print(f"  Intra-class HD (mean±std): {intra_hd.mean():.4f} ± {intra_hd.std():.4f}")
    print(f"  Inter-class HD (mean±std): {inter_hd.mean():.4f} ± {inter_hd.std():.4f}")
    print(f"  EER                      : {eer:.4f} ({eer*100:.2f}%)")
    print(f"  Threshold optimal (EER)  : HD < {eer_threshold:.4f}")
    print(f"{'='*60}")

    return {
        "intra_hd": intra_hd,
        "inter_hd": inter_hd,
        "thresholds": thresholds,
        "far_curve": far_curve,
        "frr_curve": frr_curve,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }


def plot_results(results: dict, output_dir: str = "."):
    """Menghasilkan visualisasi distribusi HD dan kurva ROC."""
    os.makedirs(output_dir, exist_ok=True)

    intra = results["intra_hd"]
    inter = results["inter_hd"]
    thresholds = results["thresholds"]
    far = results["far_curve"]
    frr = results["frr_curve"]
    eer = results["eer"]
    eer_t = results["eer_threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Distribusi HD ---
    ax1 = axes[0]
    bins = np.linspace(0, 0.7, 60)
    ax1.hist(intra, bins=bins, alpha=0.7, color="#2196F3", label="Intra-class (sama)", density=True)
    ax1.hist(inter, bins=bins, alpha=0.7, color="#F44336", label="Inter-class (beda)", density=True)
    ax1.axvline(eer_t, color="#4CAF50", linestyle="--", linewidth=2, label=f"Threshold = {eer_t:.3f}")
    ax1.set_xlabel("Hamming Distance")
    ax1.set_ylabel("Density")
    ax1.set_title("Distribusi Hamming Distance")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Plot 2: FAR vs FRR ---
    ax2 = axes[1]
    ax2.plot(thresholds, far * 100, color="#F44336", linewidth=2, label="FAR")
    ax2.plot(thresholds, frr * 100, color="#2196F3", linewidth=2, label="FRR")
    ax2.axvline(eer_t, color="#4CAF50", linestyle="--", label=f"EER = {eer*100:.2f}%")
    ax2.set_xlabel("Threshold (HD)")
    ax2.set_ylabel("Error Rate (%)")
    ax2.set_title("FAR & FRR vs Threshold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # --- Plot 3: ROC Curve ---
    ax3 = axes[2]
    ax3.plot(far * 100, (1 - frr) * 100, color="#9C27B0", linewidth=2)
    ax3.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Random (AUC=0.5)")
    ax3.scatter([eer * 100], [(1 - eer) * 100], color="#4CAF50", s=100, zorder=5, label=f"EER ({eer*100:.2f}%)")
    ax3.set_xlabel("False Acceptance Rate (%)")
    ax3.set_ylabel("Genuine Acceptance Rate (%)")
    ax3.set_title("ROC Curve")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xlim(-2, 102)
    ax3.set_ylim(-2, 102)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "benchmark_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 Plot disimpan di: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark akurasi sistem iris recognition."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path ke folder dataset (CASIA/UBIRIS).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Batasi jumlah sampel yang diproses (0 = semua). Default: 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Direktori output untuk menyimpan plot. Default: results/",
    )
    parser.add_argument(
        "--max-inter-pairs",
        type=int,
        default=5000,
        help="Batas maksimum pasangan inter-class. Default: 5000.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"❌ Direktori tidak ditemukan: {args.dataset_path}")
        print("\nPastikan dataset sudah diunduh dan di-extract.")
        print("Contoh struktur yang diharapkan:")
        print("  data/CASIA-Iris-Interval/")
        print("    ├── 001/")
        print("    │   ├── L/")
        print("    │   │   ├── S1001L01.jpg")
        print("    │   │   └── ...")
        print("    │   └── R/")
        print("    └── 002/")
        sys.exit(1)

    results = run_benchmark(
        args.dataset_path,
        max_samples=args.max_samples,
        max_inter_pairs=args.max_inter_pairs,
    )

    if results is not None:
        plot_results(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
