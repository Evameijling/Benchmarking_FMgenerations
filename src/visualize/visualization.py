import os
from pathlib import Path
import torch
import random
import rioxarray as rxr
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np

from src.utils import (
    s1rtc_thumbnail_torch, s2_thumbnail_torch, dem_thumbnail_torch, classes_thumbnail_torch
)

# Class labels for better understanding (same as visualize_single.py)
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

def render_output(modality, tensor):
    tensor = tensor.float()
    if modality == "S1RTC":
        return s1rtc_thumbnail_torch(tensor[:, 0], tensor[:, 1])
    elif modality == "S2L2A":
        return s2_thumbnail_torch(tensor[:, 3], tensor[:, 2], tensor[:, 1])
    elif modality == "DEM":
        return dem_thumbnail_torch(tensor[:, 0:1], hillshade=True)
    elif modality == "LULC":
        # Handle 10-channel LULC output from FSQ-VAE tokenizer
        if tensor.shape[1] == 10:
            # Convert 10-channel FSQ-VAE output to single-channel class indices
            class_indices = torch.argmax(tensor, dim=1, keepdim=True).float()
            print(f"Converted LULC from {tensor.shape} to {class_indices.shape}")
            print(f"Class range: {class_indices.min():.0f} - {class_indices.max():.0f}")
            return classes_thumbnail_torch(class_indices)
        else:
            # Single-channel LULC (original format)
            return classes_thumbnail_torch(tensor[:, 0:1])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

class ComparisonVisualizer:
    def __init__(self, input_modality="S1RTC", output_modality="S2L2A", root=None):
        self.root = Path(root)
        self.input_modality = input_modality
        self.output_modality = output_modality
        
        # Extract base modalities for proper handling
        self.base_input_modality = self._extract_base_modality(input_modality)
        self.base_output_modality = self._extract_base_modality(output_modality)
        
        self.input_dir = self.root / "data" / "input" / input_modality
        self.output_dir = self.root / "data" / "output" / f'{output_modality}_from_{input_modality}'
        self.vis_dir = self.root / "visualizations" / "comparisons" / f"{input_modality}_to_{output_modality}"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _extract_base_modality(self, modality):
        """Extract base modality from complex names like 'LULC_from_S1RTC' -> 'LULC'"""
        if "_from_" in modality:
            return modality.split("_from_")[0]
        return modality

    def visualize(self, n_examples=5):
        input_files = sorted(list(self.input_dir.glob("*.tif")))

        if n_examples is not None and n_examples < len(input_files):
            input_files = random.sample(input_files, n_examples)

        for input_path in input_files:
            tile_name = input_path.stem
            output_path = self.output_dir / f"{tile_name}.tif"
            if not output_path.exists():
                print(f"Output not found for {tile_name}, skipping.")
                continue

            # Load input and output using the same approach as visualize_single.py
            input_arr = rxr.open_rasterio(input_path).squeeze().values
            output_arr = rxr.open_rasterio(output_path).squeeze().values
            
            print(f"Processing {tile_name}:")
            print(f"  Input shape: {input_arr.shape}, Output shape: {output_arr.shape}")

            # Handle single-band data (like input LULC)
            if input_arr.ndim == 2:
                input_arr = input_arr[None, ...]  # Add channel dimension
            if output_arr.ndim == 2:
                output_arr = output_arr[None, ...]  # Add channel dimension

            # Convert to torch tensors and add batch dimension
            input_tensor = torch.tensor(input_arr).float().unsqueeze(0)  # [1, C, H, W]
            output_tensor = torch.tensor(output_arr).float().unsqueeze(0)  # [1, C, H, W]

            # Apply scaling based on modality (same as original but with base modality)
            if self.base_output_modality == "S1RTC":
                # Convert from dB back to linear power values for visualization
                output_tensor = 10 ** (output_tensor / 10)

            # Generate thumbnails using base modalities
            try:
                input_vis = render_output(self.base_input_modality, input_tensor)
                output_vis = render_output(self.base_output_modality, output_tensor)
                
                print(f"  Input vis shape: {input_vis.shape}, Output vis shape: {output_vis.shape}")
            except Exception as e:
                print(f"Error generating thumbnails for {tile_name}: {e}")
                continue

            # Remove batch dimension if present (same as visualize_single.py)
            if input_vis.ndim == 4:
                input_vis = input_vis.squeeze(0)
            if output_vis.ndim == 4:
                output_vis = output_vis.squeeze(0)

            # Convert from CHW to HWC for matplotlib if needed (same as visualize_single.py)
            if input_vis.ndim == 3 and input_vis.shape[0] in [1, 3]:
                input_vis = input_vis.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
            if output_vis.ndim == 3 and output_vis.shape[0] in [1, 3]:
                output_vis = output_vis.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

            # Ensure both images have the same height for side-by-side comparison
            target_h = min(input_vis.shape[0], output_vis.shape[0])
            target_w_input = int(input_vis.shape[1] * target_h / input_vis.shape[0])
            target_w_output = int(output_vis.shape[1] * target_h / output_vis.shape[0])

            # Resize to target height while maintaining aspect ratio
            if input_vis.shape[:2] != (target_h, target_w_input):
                input_vis = TF.resize(input_vis.permute(2, 0, 1), (target_h, target_w_input)).permute(1, 2, 0)
            if output_vis.shape[:2] != (target_h, target_w_output):
                output_vis = TF.resize(output_vis.permute(2, 0, 1), (target_h, target_w_output)).permute(1, 2, 0)

            # Plot side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Handle different image formats (same as visualize_single.py)
            def plot_image(ax, img, title):
                if img.ndim == 2:
                    # Grayscale [H, W]
                    ax.imshow(img.cpu().numpy(), cmap='gray')
                elif img.ndim == 3 and img.shape[-1] == 1:
                    # Grayscale with channel dimension [H, W, 1]
                    ax.imshow(img.squeeze(-1).cpu().numpy(), cmap='gray')
                else:
                    # RGB [H, W, 3]
                    ax.imshow(img.cpu().numpy())
                ax.set_title(title)
                ax.axis('off')

            plot_image(ax1, input_vis, f"{self.base_input_modality} Input")
            plot_image(ax2, output_vis, f"Generated {self.base_output_modality}")
            
            plt.suptitle(f"{self.input_modality} → {self.output_modality}: {tile_name}")
            plt.tight_layout()
            
            vis_path = self.vis_dir / f"{tile_name}.png"
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"  Saved comparison: {vis_path}")

            # Print class distribution for LULC modalities (same as visualize_single.py)
            if self.base_input_modality == "LULC":
                self._print_lulc_stats("Input", input_arr.squeeze())
            if self.base_output_modality == "LULC":
                self._print_lulc_stats("Output", output_arr.squeeze())

    def _print_lulc_stats(self, label, arr):
        """Print LULC class distribution (same logic as visualize_single.py)"""
        # Get class data for statistics
        if arr.ndim == 3 and arr.shape[0] == 10:
            # Use argmax for 10-channel data
            class_data = arr.argmax(axis=0)
            print(f"  {label} LULC: Used argmax for class distribution")
        elif arr.ndim == 2:
            # Single-channel data
            class_data = arr
        else:
            # Multi-channel but not 10 - use first channel
            class_data = arr[0] if arr.ndim > 2 else arr

        # Print top 3 classes only (to keep output manageable)
        unique_vals, counts = np.unique(class_data, return_counts=True)
        total_pixels = class_data.size
        
        # Show top 3 most common classes
        top_indices = np.argsort(counts)[::-1][:3]
        print(f"  {label} LULC top 3 classes:")
        for i, idx in enumerate(top_indices, 1):
            class_id = int(unique_vals[idx])
            percentage = (counts[idx] / total_pixels) * 100
            class_name = LULC_CLASS_NAMES.get(class_id, f"Unknown Class {class_id}")
            print(f"    {i}. Class {class_id} ({class_name}): {percentage:.1f}%")

if __name__ == "__main__":
    input_modality = "S1RTC"
    output_modality = "LULC"

    visualizer = ComparisonVisualizer(
        input_modality=input_modality,
        output_modality=output_modality,
        root=Path("/home/egm/Data/Projects/CopGen")
    )
    visualizer.visualize(n_examples=5)