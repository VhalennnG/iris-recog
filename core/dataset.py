import os
import glob
import cv2
import numpy as np
from typing import List, Tuple, Dict

class IrisDataset:
    """
    Lazy-loading dataset pipeline untuk CASIA-Iris / UBIRIS.
    Asumsi struktur dataset: /dataset_path/subject_id/mata_kiri_atau_kanan/image.jpg
    """
    def __init__(self, dataset_dir: str, valid_extensions: Tuple[str, ...] = ('.jpg', '.bmp', '.tiff')):
        self.dataset_dir = dataset_dir
        self.valid_extensions = valid_extensions
        self.image_paths: List[str] = self._load_paths()
        self.subject_ids: List[str] = self._extract_subject_ids()

    def _load_paths(self) -> List[str]:
        paths = []
        for ext in self.valid_extensions:
            # Mencari rekursif ke dalam sub-folder
            paths.extend(glob.glob(os.path.join(self.dataset_dir, f"**/*{ext}"), recursive=True))
        return sorted(paths)

    def _extract_subject_ids(self) -> List[str]:
        # Ekstrak ID Subjek dari path (disesuaikan dengan hierarki folder CASIA)
        # Asumsi folder level pertama setelah dataset_dir adalah subject_id
        return [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Returns citra dalam bentuk NumPy array dan ID subjeknya."""
        path = self.image_paths[idx]
        subject_id = self.subject_ids[idx]
        
        # Baca secara grayscale karena iris teksturnya cukup dengan intensitas cahaya
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Gagal membaca citra di path: {path}")
            
        return image, subject_id

    def split_dataset(self, test_size: float = 0.2) -> Tuple['IrisDataset', 'IrisDataset']:
        """
        Pemisahan data berbasis Subject ID untuk mencegah Data Leakage.
        """
        unique_subjects = list(set(self.subject_ids))
        np.random.shuffle(unique_subjects)
        
        split_idx = int(len(unique_subjects) * (1 - test_size))
        train_subjects = set(unique_subjects[:split_idx])
        
        train_paths = [p for p, s in zip(self.image_paths, self.subject_ids) if s in train_subjects]
        test_paths = [p for p, s in zip(self.image_paths, self.subject_ids) if s not in train_subjects]
        
        # Membuat instansiasi baru untuk Train dan Test
        train_ds = IrisDataset(self.dataset_dir)
        train_ds.image_paths, train_ds.subject_ids = train_paths, [s for s in self.subject_ids if s in train_subjects]
        
        test_ds = IrisDataset(self.dataset_dir)
        test_ds.image_paths, test_ds.subject_ids = test_paths, [s for s in self.subject_ids if s not in train_subjects]
        
        return train_ds, test_ds