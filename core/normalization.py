import cv2
import numpy as np
from typing import Tuple

def unwrap_iris(
    image: np.ndarray, 
    pupil: Tuple[int, int, int], 
    iris: Tuple[int, int, int], 
    width: int = 512, 
    height: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transformasi Daugman's Rubber Sheet Model.
    Mengubah area iris menjadi persegi panjang (height x width).
    """
    xp, yp, rp = pupil
    xi, yi, ri = iris

    # Buat grid untuk r (0 hingga 1) dan theta (0 hingga 2*pi)
    theta = np.linspace(0, 2 * np.pi, width)
    r = np.linspace(0, 1, height)
    
    # Vektorisasi koordinat batas pupil dan iris pada setiap sudut theta
    # Ditambahkan [np.newaxis, :] untuk broadcasting dengan array 'r'
    x_pupil = xp + rp * np.cos(theta)[np.newaxis, :]
    y_pupil = yp + rp * np.sin(theta)[np.newaxis, :]
    
    x_iris = xi + ri * np.cos(theta)[np.newaxis, :]
    y_iris = yi + ri * np.sin(theta)[np.newaxis, :]
    
    # Broadcast 'r' array untuk kalkulasi grid penuh
    r_grid = r[:, np.newaxis]
    
    # Hitung koordinat Kartesian (x, y) untuk pemetaan
    x_map = (1 - r_grid) * x_pupil + r_grid * x_iris
    y_map = (1 - r_grid) * y_pupil + r_grid * y_iris
    
    # Konversi ke float32 sesuai kebutuhan cv2.remap
    x_map = np.float32(x_map)
    y_map = np.float32(y_map)

    # Lakukan interpolasi bilinear
    normalized_iris = cv2.remap(image, x_map, y_map, cv2.INTER_LINEAR)
    
    # Buat noise mask dasar (inisialisasi dengan 1, artinya valid)
    # Area kelopak mata (occlusion) bisa disisipkan ke mask ini nanti
    noise_mask = np.ones((height, width), dtype=np.uint8)

    return normalized_iris, noise_mask