import cv2
import numpy as np
from typing import Tuple, Dict, Optional

def find_pupil(image: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Mendeteksi batas pupil-iris (Inner Boundary).
    Returns: (x, y, radius) atau None jika gagal.
    """
    # Menggunakan morphological closing untuk menghilangkan noise (seperti pantulan cahaya di pupil)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # CHT untuk mendeteksi pupil (biasanya radius 10 hingga 80 pixel pada CASIA)
    circles = cv2.HoughCircles(
        morphed, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
        param1=50, param2=30, minRadius=10, maxRadius=80
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Asumsi pupil adalah lingkaran paling kuat yang terdeteksi
        return tuple(circles[0])
    return None

def find_iris_outer_boundary(image: np.ndarray, pupil_center: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
    """
    Mendeteksi batas iris-sklera (Outer Boundary).
    Returns: (x, y, radius) atau None.
    """
    # Fokus pencarian di sekitar pusat pupil
    px, py = pupil_center
    
    # Deteksi tepi menggunakan Canny
    edges = cv2.Canny(image, 10, 50)
    
    # CHT untuk iris (radius biasanya lebih besar, 80 hingga 250 pixel)
    # minDist diset kecil karena kita membatasi pencarian berdasarkan kedekatan dengan pupil
    circles = cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=40, minRadius=80, maxRadius=250
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Filter lingkaran yang pusatnya tidak terlalu jauh dari pupil (toleransi konsentris)
        valid_circles = [c for c in circles if np.sqrt((c[0]-px)**2 + (c[1]-py)**2) < 30]
        
        if valid_circles:
            # Ambil lingkaran dengan radius terbesar sebagai batas luar
            best_circle = max(valid_circles, key=lambda c: c[2])
            return tuple(best_circle)
            
    return None

def segment_iris(image: np.ndarray) -> Dict[str, Tuple[int, int, int]]:
    """Pipeline penuh untuk segmentasi mata."""
    pupil = find_pupil(image)
    if not pupil:
        raise ValueError("Pupil tidak terdeteksi.")
        
    iris = find_iris_outer_boundary(image, (pupil[0], pupil[1]))
    if not iris:
        raise ValueError("Batas Iris tidak terdeteksi.")
        
    return {"pupil": pupil, "iris": iris}