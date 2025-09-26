import os
from pathlib import Path
import torch
import random
import rioxarray as rxr
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from src.utils import (
    s1rtc_thumbnail_torch, s2_thumbnail_torch, dem_thumbnail_torch, classes_thumbnail_torch
)

def render_output(modality, tensor):
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

class ComparisonVisualizer:
    def __init__(self, input_modality="S1RTC", output_modality="S2L2A", root=None):
        self.root = Path(root)
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.input_dir = self.root / "data" / "input" / input_modality
        self.output_dir = self.root / "data" / "output" / output_modality
        self.vis_dir = self.root / "visualizations" / "comparisons" / f"{input_modality}_to_{output_modality}"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

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

            # Load input and output
            input_arr = rxr.open_rasterio(input_path).squeeze().values  # [C, H, W]
            output_arr = rxr.open_rasterio(output_path).squeeze().values  # [C, H, W]

            # Convert to torch tensors (float) and add batch dimension
            input_tensor = torch.tensor(input_arr).float().unsqueeze(0)
            output_tensor = torch.tensor(output_arr).float().unsqueeze(0)

            # # Apply scaling based on modality
            # if self.output_modality == "S2L2A":
            #     # Convert from 0-10000 back to 0-1 range for visualization
            #     output_tensor = output_tensor / 1000.0
            if self.output_modality == "S1RTC":
                # Convert from dB back to linear power values for visualization
                # The thumbnail function expects linear power values, not dB
                output_tensor = 10 ** (output_tensor / 10)
                min_val = output_tensor.min()
                max_val = output_tensor.max()
                # print(f"S1RTC converted to linear power range: {min_val:.6f} to {max_val:.6f}")

            # Generate thumbnails
            input_vis = render_output(self.input_modality, input_tensor)
            output_vis = render_output(self.output_modality, output_tensor)

            # Convert to channel-first if needed
            if input_vis.ndim == 4:
                input_vis = input_vis.permute(0, 3, 1, 2)
            if output_vis.ndim == 4:
                output_vis = output_vis.permute(0, 3, 1, 2)

            # Ensure both are [B, C, H, W]
            def ensure_bchw(t):
                if t.ndim == 3:
                    return t.unsqueeze(0)
                elif t.ndim == 4:
                    return t
                else:
                    raise ValueError("Unexpected tensor shape")

            input_vis = ensure_bchw(input_vis)
            output_vis = ensure_bchw(output_vis)

            # Resize to smallest H, W
            target_h = min(input_vis.shape[2], output_vis.shape[2])
            target_w = min(input_vis.shape[3], output_vis.shape[3])
            input_vis_resized = F.interpolate(input_vis, size=(target_h, target_w), mode="bilinear", align_corners=False)
            output_vis_resized = F.interpolate(output_vis, size=(target_h, target_w), mode="bilinear", align_corners=False)

            # Plot side by side
            grid = vutils.make_grid(torch.cat([input_vis_resized, output_vis_resized], dim=0), nrow=2)
            plt.figure(figsize=(8, 4))
            plt.imshow(TF.to_pil_image(grid))
            plt.axis('off')
            plt.title(f"{self.input_modality} Input vs Generated {self.output_modality}\n{tile_name}")
            vis_path = self.vis_dir / f"{tile_name}.png"
            plt.savefig(vis_path)
            plt.close()
            print(f"Saved visualization: {vis_path}")

if __name__ == "__main__":
    input_modality = "S1RTC"
    output_modality = "S2L2A"

    visualizer = ComparisonVisualizer(
        input_modality=input_modality,
        output_modality=output_modality,
        root=Path("/home/egm/Data/Projects/CopGen")
    )
    visualizer.visualize(n_examples=5)