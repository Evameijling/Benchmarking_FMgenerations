import torch
import rioxarray as rxr
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path
from .base import render_output, print_lulc_stats

class Visualizer:
    def __init__(self, InOutput: str, modality: str, tile: str, root: Path, crop_size=256):
        self.InOutput = InOutput
        self.tile = tile
        self.root = Path(root)
        self.crop_size = crop_size  # Add crop_size parameter

        if "_from_" in modality:
            self.base_modality = modality.split("_from_")[0]
            self.full_modality_path = modality
        else:
            self.base_modality = modality
            self.full_modality_path = modality

        self.data_dir = self.root / "data" / InOutput / self.full_modality_path
        self.tif_path = self._find_tile_file()

        self.vis_dir = self.root / "visualizations" / "singles" / InOutput / self.full_modality_path
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _find_tile_file(self):
        matches = []
        for f in self.data_dir.glob("*.tif"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                tile_base = "_".join(parts[:2])
                if tile_base == self.tile:
                    matches.append(f)

        if not matches:
            raise FileNotFoundError(f"No file with tile base '{self.tile}' found in {self.data_dir}")
        if len(matches) > 1:
            print(f"WARNING: Multiple matches for tile '{self.tile}': {[f.name for f in matches]}")
        
        # print file name match
        print(f"Found file: {matches[0].name}")
        return matches[0]

    def visualize(self, save=True, show=False):
        input_arr = rxr.open_rasterio(self.tif_path).squeeze().values
        original_arr = input_arr.copy()
        if input_arr.ndim == 2:
            input_arr = input_arr[None, ...]

        input_tensor = torch.tensor(input_arr).float().unsqueeze(0)
        
        # Apply crop if this is input data (not output data)
        if self.InOutput == "input":
            crop = T.CenterCrop(self.crop_size)
            input_tensor = crop(input_tensor)
            title_suffix = f" ({self.crop_size}x{self.crop_size})"
        else:
            title_suffix = ""
        
        input_vis = render_output(self.base_modality, input_tensor)

        if input_vis.ndim == 4:
            input_vis = input_vis.squeeze(0)
        if input_vis.ndim == 3 and input_vis.shape[0] in [1, 3]:
            input_vis = input_vis.permute(1, 2, 0)

        plt.figure(figsize=(10, 8))
        plt.imshow(input_vis.cpu().numpy())
        plt.axis("off")
        plt.title(f"{self.base_modality} - {self.tif_path.stem}{title_suffix}")

        if save:
            vis_path = self.vis_dir / f"{self.tif_path.stem}.png"
            plt.savefig(vis_path, bbox_inches="tight", dpi=150)
            print(f"Saved visualization: {vis_path}")
        if show:
            plt.show()
        plt.close()

        if self.base_modality == "LULC":
            if self.InOutput == "input":
                # Use cropped data for stats
                cropped_arr = crop(torch.tensor(original_arr).unsqueeze(0) if original_arr.ndim == 2 else torch.tensor(original_arr[None, ...]).unsqueeze(0))
                print_lulc_stats("Single (cropped)", cropped_arr.squeeze().numpy())
            else:
                print_lulc_stats("Single", original_arr)