import torch
import rioxarray as rxr
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from src.utils import (
    s1rtc_thumbnail_torch, s2_thumbnail_torch, dem_thumbnail_torch, classes_thumbnail_torch
)

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

class Visualizer:
    def __init__(self, InOutput: str, modality: str, tile: str, root: Path = None):
        """
        Args:
            InOutput: "input" or "output"
            modality: "LULC", "S1RTC", "S2L2A", "DEM" or "LULC_from_S1RTC", etc.
            tile: Tile identifier (e.g., "433U_186R")
            root: Root directory path
        """
        self.InOutput = InOutput
        self.tile = tile
        
        # Extract base modality if "_from_" is in the modality string
        if "_from_" in modality:
            # Extract base modality from "LULC_from_S1RTC" -> "LULC"
            self.base_modality = modality.split("_from_")[0]
            self.full_modality_path = modality  # Keep full path for directory
            print(f"Debug: Extracted base modality '{self.base_modality}' from '{modality}'")
        else:
            self.base_modality = modality
            self.full_modality_path = modality

        # Set modality for backward compatibility
        self.modality = self.base_modality

        # Visualization directory relative to tif file
        self.root = Path(root)
        
        # Handle directory structure
        if "_from_" in modality:
            # For generated files like "LULC_from_S1RTC"
            self.data_dir = self.root / "data" / InOutput / self.full_modality_path
        else:
            # For input files like "LULC"
            self.data_dir = self.root / "data" / InOutput / self.base_modality
        
        print(f"Debug: Looking for files in: {self.data_dir}")
        print(f"Debug: Directory exists: {self.data_dir.exists()}")
        print(f"Debug: Using modality: {self.modality}")
        if self.data_dir.exists():
            print(f"Debug: Files in directory: {list(self.data_dir.glob('*.tif'))}")

        # Find the actual file that contains the tile name
        self.tif_path = self._find_tile_file()
        
        if "_from_" in modality:
            # Mirror the data structure: visualizations/singles/output/LULC_from_S1RTC/
            self.vis_dir = self.root / "visualizations" / "singles" / InOutput / self.full_modality_path
        else:
            # Simple structure: visualizations/singles/output/LULC/
            self.vis_dir = self.root / "visualizations" / "singles" / InOutput / self.modality
        
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _find_tile_file(self):
        """Find the TIF file that contains the tile identifier"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Look for files that contain the tile name
        matching_files = list(self.data_dir.glob(f"*{self.tile}*.tif"))
        
        if not matching_files:
            raise FileNotFoundError(f"No TIF file found containing '{self.tile}' in {self.data_dir}")
        
        if len(matching_files) > 1:
            print(f"Warning: Multiple files found for tile '{self.tile}': {[f.name for f in matching_files]}")
            print(f"Using the first one: {matching_files[0].name}")
        
        return matching_files[0]

    def visualize(self, save=True, show=False):
        if not self.tif_path.exists():
            raise FileNotFoundError(f"TIF not found: {self.tif_path}")

        # Load tif as array [C, H, W] - rioxarray returns channel-first format
        input_arr = rxr.open_rasterio(self.tif_path).squeeze().values
        print(f"Loaded {self.tif_path}")
        print(f"Loaded shape: {input_arr.shape}, min={input_arr.min()}, max={input_arr.max()}")

        # Store original data for LULC statistics
        original_arr = input_arr.copy()

        # Handle single-band data (like input LULC)
        if input_arr.ndim == 2:
            input_arr = input_arr[None, ...]  # Add channel dimension

        # Convert to torch tensor [1, C, H, W] - no transpose needed!
        input_tensor = torch.tensor(input_arr).float().unsqueeze(0)

        # Generate thumbnail
        input_vis = render_output(self.modality, input_tensor)
        
        print(f"input_vis shape after render_output: {input_vis.shape}")
        
        # Remove batch dimension if present
        if input_vis.ndim == 4:
            input_vis = input_vis.squeeze(0)  # Remove batch dimension -> [C, H, W] or [H, W, C]
        
        print(f"input_vis shape after squeeze: {input_vis.shape}")
        
        # Convert from CHW to HWC for matplotlib if needed
        if input_vis.ndim == 3:
            if input_vis.shape[0] in [1, 3]:  # Channel first format [C, H, W]
                input_vis = input_vis.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                print(f"Permuted to HWC: {input_vis.shape}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        print(f"Final input_vis shape for plotting: {input_vis.shape}")
        
        # Handle different image formats
        if input_vis.ndim == 2:
            # Grayscale [H, W]
            plt.imshow(input_vis.cpu().numpy(), cmap='viridis' if self.modality == 'LULC' else 'gray')
        elif input_vis.ndim == 3 and input_vis.shape[-1] == 1:
            # Grayscale with channel dimension [H, W, 1]
            input_vis = input_vis.squeeze(-1)
            plt.imshow(input_vis.cpu().numpy(), cmap='viridis' if self.modality == 'LULC' else 'gray')
        else:
            # RGB [H, W, 3]
            plt.imshow(input_vis.cpu().numpy())

        plt.axis('off')
        plt.title(f"{self.modality} - {self.tif_path.stem}")
        if save:
            vis_path = self.vis_dir / f"{self.tif_path.stem}.png"
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization: {vis_path}")
        if show:
            plt.show()
        plt.close()

        # Print class distribution for LULC
        if self.modality == "LULC":
            # Get class data for statistics
            if original_arr.ndim == 3 and original_arr.shape[0] == 10:
                # Use argmax for 10-channel data
                class_data = original_arr.argmax(axis=0)
                print("Used argmax for class distribution analysis")
            elif original_arr.ndim == 2:
                # Single-channel data
                class_data = original_arr
            else:
                # Multi-channel but not 10 - use first channel
                class_data = original_arr[0] if original_arr.ndim > 2 else original_arr

            # Print class distribution
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
    ROOT = Path("/home/egm/Data/Projects/CopGen")

    # Example: Visualize generated LULC from S1RTC
    InOutput = "output/LULC_from_S1RTC"  # [input, output, output/LULC_from_S1RTC, etc.]
    modality = "LULC" # [DEM, LULC, S1RTC, S2L2A]
    tile = "433U_1061L"
    visualizer = Visualizer(
        InOutput=InOutput,
        modality=modality, 
        tile=tile,
        root=ROOT
    )
    visualizer.visualize(save=True, show=False)