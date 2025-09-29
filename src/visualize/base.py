import torch
import numpy as np
from src.utils import (
    s1rtc_thumbnail_torch,
    s2_thumbnail_torch,
    dem_thumbnail_torch,
    classes_thumbnail_torch,
)

LULC_CLASS_NAMES = {
    0: "None/NoData",
    1: "Water",
    2: "Trees",
    3: "Flooded vegetation",
    4: "Crops",
    5: "Built Area",
    6: "Bare ground",
    7: "Snow/Ice",
    8: "Clouds",
    9: "Rangeland",
}

def render_output(modality, tensor: torch.Tensor):
    tensor = tensor.float()
    if modality == "S1RTC":
        return s1rtc_thumbnail_torch(tensor[:, 0], tensor[:, 1])
    elif modality == "S2L2A":
        return s2_thumbnail_torch(tensor[:, 3], tensor[:, 2], tensor[:, 1])
    elif modality == "DEM":
        return dem_thumbnail_torch(tensor[:, 0:1], hillshade=True)
    elif modality == "LULC":
        if tensor.shape[1] == 10:
            class_indices = torch.argmax(tensor, dim=1, keepdim=True).float()
            return classes_thumbnail_torch(class_indices)
        else:
            return classes_thumbnail_torch(tensor[:, 0:1])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def print_lulc_stats(label: str, arr: np.ndarray):
    if arr.ndim == 3 and arr.shape[0] == 10:
        class_data = arr.argmax(axis=0)
    elif arr.ndim == 2:
        class_data = arr
    else:
        class_data = arr[0] if arr.ndim > 2 else arr

    unique_vals, counts = np.unique(class_data, return_counts=True)
    total_pixels = class_data.size

    top_indices = np.argsort(counts)[::-1][:3]
    print(f"\n{label} LULC class distribution (top 3):")
    for i, idx in enumerate(top_indices, 1):
        class_id = int(unique_vals[idx])
        percentage = (counts[idx] / total_pixels) * 100
        class_name = LULC_CLASS_NAMES.get(class_id, f"Unknown {class_id}")
        print(f"  {i}. Class {class_id} ({class_name}): {percentage:.1f}%")
