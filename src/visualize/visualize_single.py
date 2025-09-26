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
        return classes_thumbnail_torch(tensor[:, 0:1])
    else:
        raise ValueError(f"Unsupported modality: {modality}")

class Visualizer:
    def __init__(self, InOutput: str, modality: str, tile: str, root: Path = None):
        """
        Args:
            InOutput: "input" or "output"
            modality: One of ["S1RTC", "S2L2A", "DEM", "LULC"]
            tile: Tile identifier (e.g., "433U_186R")
            root: Root directory path
        """
        self.InOutput = InOutput  # "input" or "output"
        self.tile = tile
        self.modality = modality

        # Visualization directory relative to tif file
        self.root = Path(root)
        self.data_dir = self.root / "data" / InOutput / modality
        
        # Find the actual file that contains the tile name
        self.tif_path = self._find_tile_file()
        
        self.vis_dir = self.root / "visualizations" / "singles" / self.InOutput / f"{self.modality}"
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

        # Handle single-band data (like LULC)
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

if __name__ == "__main__":
    # tif_file = "/home/egm/Data/Projects/CopGen/data/output/S2L2A/433U_183R.tif"
    # modality = "S2L2A"

    # visualizer = Visualizer(tif_path=tif_file, modality=modality, root=Path("/home/egm/Data/Projects/CopGen"))
    # visualizer.visualize(save=True, show=False)

    ROOT = Path("/home/egm/Data/Projects/CopGen")

    InOutput = "output"  # [input, output]
    modality = "S2L2A" # [DEM, LULC, S1RTC, S2L2A]
    tile = "433U_310R.tif"
    visualizer = Visualizer(
        InOutput=InOutput,  # or "output"
        modality=modality, 
        tile=tile,
        root=ROOT
    )
    visualizer.visualize(save=True, show=False)