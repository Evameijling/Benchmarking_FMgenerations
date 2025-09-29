import sys
import torch
import rioxarray as rxr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import the thumbnail function
sys.path.append(str(Path(__file__).parent.parent))
from utils.thumbnails import classes_thumbnail_torch

# Class labels for better understanding (same as visualize_single_TM.py)
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
    9: "Rangeland"
}

def plot_lulc_input_from_tif(tif_path, save_path=None):
    """
    Plot LULC input file using classes_thumbnail_torch for proper color mapping
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"TIF file not found: {tif_path}")

    # Load data
    data = rxr.open_rasterio(tif_path).squeeze().values
    print(f"Loaded {tif_path.name}, shape: {data.shape}, dtype: {data.dtype}")
    print(f"Value range: {data.min()} to {data.max()}")

    # Handle 10-channel LULC output (same logic as visualize_single_TM.py)
    if data.ndim == 3 and data.shape[0] == 10:
        print("Converting 10-channel FSQ-VAE output to class indices using argmax")
        class_data = data.argmax(axis=0)  # Convert to single channel
        print(f"Converted from {data.shape} to {class_data.shape}")
        print(f"Class range: {class_data.min()} to {class_data.max()}")
        
        # Add channel dimension for torch
        class_data = class_data[None, ...]  # [H, W] -> [1, H, W]
        input_tensor = torch.tensor(class_data).float().unsqueeze(0)  # [1, 1, H, W]
    elif data.ndim == 2:
        # Single-band data (typical for input LULC)
        data = data[None, ...]  # Add channel dimension: [H, W] -> [1, H, W]
        input_tensor = torch.tensor(data).float().unsqueeze(0)
        class_data = data[0]  # For statistics
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    print(f"Tensor shape: {input_tensor.shape}")

    # Generate RGB visualization using classes_thumbnail_torch
    rgb_vis = classes_thumbnail_torch(input_tensor[:, 0:1])  # Use the class indices
    
    # Convert from [1, 3, H, W] to [H, W, 3] for matplotlib
    rgb_vis = rgb_vis.squeeze(0).permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    
    print(f"RGB visualization shape: {rgb_vis.shape}")

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_vis.cpu().numpy())
    plt.axis('off')
    plt.title(f"LULC - {tif_path.stem}")

    # Save image
    if save_path is None:
        save_path = tif_path.with_suffix('.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved LULC visualization to {save_path}")

    # Print class distribution using the converted class indices
    unique_vals, counts = np.unique(class_data, return_counts=True)
    total_pixels = class_data.size
    
    print(f"\nClass Distribution (Total pixels: {total_pixels:,}):")
    print("-" * 60)
    
    for val, count in zip(unique_vals, counts):
        class_id = int(val)
        percentage = (count / total_pixels) * 100
        class_name = LULC_CLASS_NAMES.get(class_id, f"Unknown Class {class_id}")
        print(f"  Class {class_id:2d} ({class_name:15s}): {count:8,} pixels ({percentage:5.1f}%)")
    
    # Show top 3 most common classes
    top_indices = np.argsort(counts)[::-1][:3]
    print(f"\nTop 3 most common classes:")
    for i, idx in enumerate(top_indices, 1):
        class_id = int(unique_vals[idx])
        percentage = (counts[idx] / total_pixels) * 100
        class_name = LULC_CLASS_NAMES.get(class_id, f"Unknown Class {class_id}")
        print(f"  {i}. Class {class_id} ({class_name}): {percentage:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_single_copgen.py /path/to/lulc_input.tif")
        print("Example: python visualize_single_copgen.py /home/egm/Data/Projects/CopGen/data/input/LULC/433U_183R_2017.tif")
    else:
        plot_lulc_input_from_tif(sys.argv[1])