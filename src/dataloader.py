import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.patient_dirs = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_id = self.patient_dirs[idx]
        patient_path = os.path.join(self.data_dir, patient_id)

        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        for mod in modalities:
            file_path = os.path.join(patient_path, f"{patient_id}_{mod}.nii")
            img_data = nib.load(file_path).get_fdata().astype(np.float32)
            images.append(img_data)

        image = np.stack(images, axis=0)

        seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
        label = nib.load(seg_path).get_fdata().astype(np.float32)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label