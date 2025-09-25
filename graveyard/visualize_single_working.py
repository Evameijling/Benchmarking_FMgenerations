import torch
import rioxarray as rxr
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from pathlib import Path

from src.utils import (
    s1rtc_thumbnail_torch, s2_thumbnail_torch, dem_thumbnail_torch, classes_thumbnail_torch
)

def render_output(modality, tensor):
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

class Visualizer:
    def __init__(self, tif_path, modality):
        """
        Args:
            tif_path: Path to the single TIF file
            modality: One of ["S1RTC", "S2L2A", "DEM", "LULC"]
        """
        self.tif_path = Path(tif_path)
        self.modality = modality

        # Visualization directory relative to tif file
        project_root = self.tif_path.parent.parent.parent
        self.vis_dir = project_root / "visualizations" / f"{modality}_single"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def visualize(self, save=True, show=False):
        if not self.tif_path.exists():
            raise FileNotFoundError(f"TIF not found: {self.tif_path}")

        # Load tif as array [C, H, W] or [H, W, C]
        input_arr = rxr.open_rasterio(self.tif_path).squeeze().values
        print(f"Loaded shape before reorder: {input_arr.shape}, "
            f"min={input_arr.min()}, max={input_arr.max()}")

        if input_arr.ndim == 3:
            if input_arr.shape[0] in [1, 3, 4, 12, 13]:  # Added 12 for S2L2A
                # already channel-first (bands, H, W)
                arr_chw = input_arr
                print(f"Data is already channel-first: {arr_chw.shape}")
            else:
                # assume channel-last (H, W, C) → transpose
                arr_chw = input_arr.transpose(2, 0, 1)
                print(f"Transposed to channel-first: {arr_chw.shape}")
        else:
            raise ValueError(f"Unexpected array shape: {input_arr.shape}")

        # Tensor [1, C, H, W]
        input_tensor = torch.tensor(arr_chw).float().unsqueeze(0)

        # Generate thumbnail
        input_vis = render_output(self.modality, input_tensor)
        if input_vis.ndim == 4:
            input_vis = input_vis.permute(0, 3, 1, 2)
        if input_vis.ndim == 3:
            input_vis = input_vis.unsqueeze(0)

        # Ensure B, C, H, W
        if input_vis.ndim != 4:
            raise ValueError(f"Unexpected input_vis shape: {input_vis.shape}")

        # Plot
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Debug: Check tensor shape before plotting
        print(f"input_vis shape before squeeze: {input_vis.shape}")
        squeezed_vis = input_vis.squeeze(0)
        print(f"input_vis shape after squeeze: {squeezed_vis.shape}")
        
        # Convert to PIL and check
        pil_image = TF.to_pil_image(squeezed_vis)
        print(f"PIL image size: {pil_image.size}")
        
        plt.imshow(pil_image)
        plt.axis('off')
        plt.title(f"{self.modality} - {self.tif_path.stem}")
        if save:
            vis_path = self.vis_dir / f"{self.tif_path.stem}.png"
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization: {vis_path}")
        if show:
            plt.show()
        plt.close()

if __name__ == "__main__":
    tif_file = "/home/egm/Data/Projects/CopGen/data/input/S2L2A/433U_310R.tif"
    modality = "S2L2A"

    visualizer = Visualizer(tif_path=tif_file, modality=modality)
    visualizer.visualize(save=True, show=False)
