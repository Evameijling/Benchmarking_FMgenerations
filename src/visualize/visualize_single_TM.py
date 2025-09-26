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

def plot_lulc_from_tif(tif_path, save_path=None):
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"TIF file not found: {tif_path}")

    # Load data
    data = rxr.open_rasterio(tif_path).squeeze().values
    print(f"Loaded {tif_path.name}, shape: {data.shape}, dtype: {data.dtype}")

    # If data has more than 2D, convert logits to class indices or drop extra dims
    while len(data.shape) > 2:
        if data.shape[0] == 10:  # Possibly 10-class logits
            data = data.argmax(axis=0)
        else:
            data = data[0]

    # Plot using TerraMind-style LULC colormap
    plt.figure(figsize=(6, 6))
    plt.imshow(data, vmin=0, vmax=9, cmap=lulc_cmap, interpolation='nearest')
    plt.axis('off')
    plt.title(f"LULC - {tif_path.stem}")

    # Save image
    if save_path is None:
        save_path = tif_path.with_suffix('.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved LULC visualization to {save_path}")

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_lulc.py /path/to/lulc.tif")
    else:
        plot_lulc_from_tif(sys.argv[1])
