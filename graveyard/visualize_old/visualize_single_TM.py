import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import rioxarray as rxr
from matplotlib.colors import hex2color, LinearSegmentedColormap

# LULC color map (same as plottingutils.py)
COLORBLIND_HEX = [
    "#000000", "#3171AD", "#469C76", '#83CA70', "#EAE159",
    "#C07CB8", "#C19368", "#6FB2E4", "#F1F1F1", "#C66526"
]
COLORBLIND_RGB = [hex2color(h) for h in COLORBLIND_HEX]
lulc_cmap = LinearSegmentedColormap.from_list('lulc', COLORBLIND_RGB, N=10)

# Class labels for better understanding
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

def plot_lulc_from_tif(tif_path, save_path=None):
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"TIF file not found: {tif_path}")

    # Load data
    data = rxr.open_rasterio(tif_path).squeeze().values
    print(f"Loaded {tif_path.name}, shape: {data.shape}, dtype: {data.dtype}")
    print(f"Value range: {data.min()} to {data.max()}")

    # Store original data for statistics before processing
    original_data = data.copy()

    # If data has more than 2D, convert logits to class indices or drop extra dims
    while len(data.shape) > 2:
        if data.shape[0] == 10:  # Possibly 10-class logits
            print("Converting 10-channel logits to class indices using argmax")
            data = data.argmax(axis=0)
        else:
            print(f"Taking first channel from {data.shape}")
            data = data[0]

    print(f"Final data shape: {data.shape}")
    print(f"Final value range: {data.min()} to {data.max()}")

    # Class distribution analysis
    unique_vals, counts = np.unique(data, return_counts=True)
    total_pixels = data.size
    
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

    # Additional statistics for multi-channel input
    if len(original_data.shape) > 2:
        print(f"\nOriginal multi-channel statistics:")
        print(f"  Shape: {original_data.shape}")
        print(f"  Min per channel: {original_data.min(axis=(1,2))}")
        print(f"  Max per channel: {original_data.max(axis=(1,2))}")
        print(f"  Mean per channel: {original_data.mean(axis=(1,2))}")

    # Plot using TerraMind-style LULC colormap
    plt.figure(figsize=(8, 8))
    plt.imshow(data, vmin=0, vmax=9, cmap=lulc_cmap, interpolation='nearest')
    plt.axis('off')
    plt.title(f"LULC - {tif_path.stem}")

    # Add colorbar with class names
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_ticks(range(10))
    cbar.set_ticklabels([LULC_CLASS_NAMES[i] for i in range(10)])
    cbar.set_label('LULC Classes')

    # Save image
    if save_path is None:
        save_path = tif_path.with_suffix('.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved LULC visualization to {save_path}")

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_single_TM.py /path/to/lulc.tif")
        print("Example: python visualize_single_TM.py /home/egm/Data/Projects/CopGen/data/output/LULC_from_S1RTC/433U_1061L.tif")
    else:
        plot_lulc_from_tif(sys.argv[1])