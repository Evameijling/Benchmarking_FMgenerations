import torch
import rioxarray as rxr
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path
from .base import render_output, print_lulc_stats

class ComparisonVisualizer:
    def __init__(self, input_modality, output_modality, root):
        self.root = Path(root)
        self.input_modality = input_modality
        self.output_modality = output_modality

        self.base_input_modality = self._extract_base(input_modality)
        self.base_output_modality = self._extract_base(output_modality)

        self.input_dir = self.root / "data" / "input" / input_modality
        self.output_dir = self.root / "data" / "output" / f"{output_modality}_from_{input_modality}"
        self.vis_dir = self.root / "visualizations" / "comparisons" / f"{input_modality}_to_{output_modality}"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _extract_base(self, modality):
        return modality.split("_from_")[0] if "_from_" in modality else modality

    def visualize(self, n_examples=5):
        input_files = sorted(list(self.input_dir.glob("*.tif")))
        if n_examples and n_examples < len(input_files):
            input_files = random.sample(input_files, n_examples)

        for input_path in input_files:
            tile = input_path.stem
            output_path = self.output_dir / f"{tile}.tif"
            if not output_path.exists():
                continue

            input_arr = rxr.open_rasterio(input_path).squeeze().values
            output_arr = rxr.open_rasterio(output_path).squeeze().values

            if input_arr.ndim == 2: input_arr = input_arr[None, ...]
            if output_arr.ndim == 2: output_arr = output_arr[None, ...]

            input_tensor = torch.tensor(input_arr).float().unsqueeze(0)
            output_tensor = torch.tensor(output_arr).float().unsqueeze(0)

            input_vis = render_output(self.base_input_modality, input_tensor)
            output_vis = render_output(self.base_output_modality, output_tensor)

            if input_vis.ndim == 4: input_vis = input_vis.squeeze(0)
            if output_vis.ndim == 4: output_vis = output_vis.squeeze(0)

            if input_vis.ndim == 3 and input_vis.shape[0] in [1, 3]:
                input_vis = input_vis.permute(1, 2, 0)
            if output_vis.ndim == 3 and output_vis.shape[0] in [1, 3]:
                output_vis = output_vis.permute(1, 2, 0)

            target_h = min(input_vis.shape[0], output_vis.shape[0])
            input_vis = TF.resize(input_vis.permute(2, 0, 1), (target_h, int(input_vis.shape[1]*target_h/input_vis.shape[0]))).permute(1, 2, 0)
            output_vis = TF.resize(output_vis.permute(2, 0, 1), (target_h, int(output_vis.shape[1]*target_h/output_vis.shape[0]))).permute(1, 2, 0)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(input_vis.cpu().numpy()); ax1.set_title(f"{self.base_input_modality} Input"); ax1.axis("off")
            ax2.imshow(output_vis.cpu().numpy()); ax2.set_title(f"Generated {self.base_output_modality}"); ax2.axis("off")
            plt.suptitle(f"{self.input_modality} → {self.output_modality}: {tile}")
            plt.tight_layout()

            vis_path = self.vis_dir / f"{tile}.png"
            plt.savefig(vis_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"Saved comparison: {vis_path}")

            if self.base_input_modality == "LULC":
                print_lulc_stats("Input", input_arr.squeeze())
            if self.base_output_modality == "LULC":
                print_lulc_stats("Output", output_arr.squeeze())
