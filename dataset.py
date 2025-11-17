import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math


def _assert_image_loaded(image, path):
    if image is None:
        raise FileNotFoundError(f'Unable to read image at {path}')


def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _assert_image_loaded(xray, path)
    xray = xray.astype(np.float32) / 255.0
    xray = xray.reshape((1, *xray.shape))
    return torch.from_numpy(xray)


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _assert_image_loaded(mask, path)
    mask = (mask > 0).astype(np.float32)
    mask = mask.reshape((1, *mask.shape))
    return torch.from_numpy(mask)


def _has_valid_path(path):
    if isinstance(path, str):
        return path.strip() != ''
    if path is None:
        return False
    if isinstance(path, float):
        return not math.isnan(path)
    return True


class Knee_dataset(Dataset):
    def __init__(self, df, return_mask: bool = True):
        self.df = df.reset_index(drop=True)
        self.return_mask = return_mask

        if self.return_mask:
            missing_idx = [
                idx for idx, path in enumerate(self.df['masks'])
                if not _has_valid_path(path)
            ]
            if missing_idx:
                raise ValueError(
                    f'Mask path missing for {len(missing_idx)} samples '
                    f'while return_mask=True. '
                    f'Example row index: {missing_idx[0]}'
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.df['xrays'].iloc[index]
        image = read_xray(img_path)

        sample = {
            'image': image,
            'image_path': img_path
        }

        if self.return_mask:
            mask_path = self.df['masks'].iloc[index]
            if not _has_valid_path(mask_path):
                raise ValueError(f'Mask required but missing for index {index}')
            mask = read_mask(str(mask_path))
            sample['mask'] = mask
            sample['mask_path'] = mask_path

        return sample
