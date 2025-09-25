import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import rioxarray as rxr
import matplotlib.pyplot as plt

from src.utils import (
    s1rtc_thumbnail_torch, s2_thumbnail_torch, dem_thumbnail_torch, classes_thumbnail_torch
)

def render_output(modality, tensor):
    """Render a tensor based on its modality"""
    # Ensure tensor is float for all operations
    tensor = tensor.float()
    
    if modality == "S1RTC":
        return s1rtc_thumbnail_torch(tensor[:, 0], tensor[:, 1])
    elif modality == "S2L2A":
        return s2_thumbnail_torch(tensor[:, 3], tensor[:, 2], tensor[:, 1])
    elif modality == "DEM":
        return dem_thumbnail_torch(tensor[:, 0:1], hillshade=True)
    elif modality == "LULC":
        return classes_thumbnail_torch(tensor)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def visualize_single_tif(tif_path, modality, output_path=None):
    """
    Visualize a single TIF file
    
    Args:
        tif_path: Path to the TIF file
        modality: Type of data ("S1RTC", "S2L2A", "DEM", "LULC")
        output_path: Optional path to save the visualization (if None, displays only)
    """
    tif_path = Path(tif_path)
    
    # Load the TIF file
    data_arr = rxr.open_rasterio(tif_path).squeeze().values  # [C, H, W]
    
    # Convert to torch tensor and add batch dimension
    tensor = torch.tensor(data_arr).float().unsqueeze(0)  # [1, C, H, W]
    
    # Debug: Print original data statistics
    print(f"Original data shape: {tensor.shape}")
    print(f"Original data range: {tensor.min():.4f} to {tensor.max():.4f}")
    print(f"Original data mean: {tensor.mean():.4f}")
    print(f"Non-zero values: {(tensor != 0).sum().item()} / {tensor.numel()}")
    
    # Apply scaling based on modality
    if modality == "S2L2A":
        # Convert from 0-10000+ range to 0-1 range for visualization
        tensor = tensor / 10000.0
        # Clamp to 0-1 range since some values might be > 10000
        # tensor = torch.clamp(tensor, 0, 1)
        print(f"S2L2A scaled and clamped to range: {tensor.min():.4f} to {tensor.max():.4f}")
    elif modality == "S1RTC":
        # Convert from dB back to linear power values for visualization
        tensor = 10 ** (tensor / 10)
        print(f"S1RTC converted to linear power range: {tensor.min():.6f} to {tensor.max():.6f}")
    
    # Generate visualization
    vis_tensor = render_output(modality, tensor)
    
    # Debug: Check visualization tensor
    print(f"Visualization tensor range: {vis_tensor.min():.4f} to {vis_tensor.max():.4f}")
    
    # Convert to PIL image format if needed
    if vis_tensor.ndim == 4 and vis_tensor.shape[0] == 1:
        vis_tensor = vis_tensor.squeeze(0)  # Remove batch dimension
    if vis_tensor.ndim == 3 and vis_tensor.shape[0] <= 3:
        vis_tensor = vis_tensor.permute(1, 2, 0)  # CHW -> HWC
    
    # Convert to numpy for matplotlib
    vis_numpy = vis_tensor.detach().cpu().numpy()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    if vis_numpy.ndim == 3:
        plt.imshow(vis_numpy)
    else:
        plt.imshow(vis_numpy, cmap='viridis')
    
    plt.title(f"{modality} - {tif_path.name}")
    plt.axis('off')
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Example usage - try S2L2A
    tif_file = "/home/egm/Data/Projects/CopGen/data/input/S2L2A/92U_5R.tif"  # Update path as needed
    modality = "S2L2A"  # Change to "S1RTC", "DEM", or "LULC" as needed
    
    # # Just display
    # visualize_single_tif(tif_file, modality)
    
    # Or save to file
    output_file = "/home/egm/Data/Projects/CopGen/visualizations/single_s2l2a_viz.png"
    visualize_single_tif(tif_file, modality, output_file)