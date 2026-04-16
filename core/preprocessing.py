import cv2
import numpy as np

def enhance_iris_visibility(
    image: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_grid_size: tuple[int, int] = (8, 8),
    bilateral_d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Meningkatkan visibilitas tekstur iris dan mengurangi noise (edge-preserving).
    
    Args:
        image: Citra mentah (Grayscale atau BGR).
        clip_limit: Threshold untuk contrast limiting pada CLAHE.
        tile_grid_size: Ukuran grid untuk histogram equalization.
        bilateral_d: Diameter pixel neighborhood.
        sigma_color: Filter sigma di color space.
        sigma_space: Filter sigma di coordinate space.
        
    Returns:
        Citra iris yang sudah ditingkatkan kualitasnya (Grayscale NumPy array).
    """
    # 1. Pastikan citra dalam format Grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # 2. Contrast Enhancement menggunakan CLAHE
    # Membantu menonjolkan fitur tekstur (crypts, furrows) pada mata gelap
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_contrast = clahe.apply(gray_image)

    # 3. Noise Reduction menggunakan Bilateral Filter
    # Menjaga ketajaman garis tepi pupil dan sklera untuk tahap Segmentasi
    filtered_image = cv2.bilateralFilter(
        enhanced_contrast, 
        d=bilateral_d, 
        sigmaColor=sigma_color, 
        sigmaSpace=sigma_space
    )

    return filtered_image