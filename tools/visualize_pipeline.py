import cv2
import matplotlib.pyplot as plt
import sys
import os

# Menambahkan root project ke sys.path agar bisa import modul core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessing import enhance_iris_visibility
from core.segmentation import segment_iris
from core.normalization import unwrap_iris

def debug_iris_pipeline(image_path: str):
    print(f"Memproses citra: {image_path}")
    raw_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if raw_img is None:
        print("Gagal memuat gambar.")
        return

    # 1. Preprocessing
    enhanced = enhance_iris_visibility(raw_img)
    
    # 2. Segmentasi
    try:
        boundaries = segment_iris(enhanced)
        p_x, p_y, p_r = boundaries['pupil']
        i_x, i_y, i_r = boundaries['iris']
    except ValueError as e:
        print(f"Segmentasi Gagal: {e}")
        return

    # 3. Normalisasi
    unwrapped, _ = unwrap_iris(enhanced, boundaries['pupil'], boundaries['iris'])

    # -- VISUALISASI --
    canvas = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    # Gambar Pupil (Merah)
    cv2.circle(canvas, (p_x, p_y), p_r, (0, 0, 255), 2)
    # Gambar Iris (Hijau)
    cv2.circle(canvas, (i_x, i_y), i_r, (0, 255, 0), 2)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Deteksi Batas (Pupil & Iris)")
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Unwrapped Iris (64 x 512)")
    plt.imshow(unwrapped, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Contoh penggunaan: uv run tools/visualize_pipeline.py path/to/eye.jpg
    if len(sys.argv) > 1:
        debug_iris_pipeline(sys.argv[1])
    else:
        print("Harap sertakan path citra. Format: uv run visualize_pipeline.py <image_path>")