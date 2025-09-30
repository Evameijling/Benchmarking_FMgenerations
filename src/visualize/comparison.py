import torch
import rioxarray as rxr
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Union, List
from .base import render_output, print_lulc_stats

class ComparisonVisualizer:
    def __init__(self, input_modality: Union[str, List[str]], output_modality: str, root, crop_size=256):
        self.root = Path(root)
        self.crop_size = crop_size  # Add crop_size parameter
        
        # Handle both single string and list inputs
        if isinstance(input_modality, str):
            self.input_modalities = [input_modality]
            self.input_modality = input_modality
        else:
            self.input_modalities = input_modality
            self.input_modality = "_".join(input_modality)
        
        self.output_modality = output_modality

        self.base_input_modality = self._extract_base(self.input_modality)
        self.base_output_modality = self._extract_base(output_modality)

        # Set up directories
        self.input_dirs = {mod: self.root / "data" / "input" / mod for mod in self.input_modalities}
        self.output_dir = self.root / "data" / "output" / f"{output_modality}_from_{self.input_modality}"
        self.vis_dir = self.root / "visualizations" / "comparisons" / f"{self.input_modality}_to_{output_modality}"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _extract_base(self, modality):
        return modality.split("_from_")[0] if "_from_" in modality else modality

    def extract_base_name(self, filename):
        """Extract base tile name, removing year suffixes"""
        base = filename
        if '_20' in base:  # Remove _2020, _2021, etc.
            base = base.split('_20')[0]
        return base

    def find_matching_file(self, directory, base_tile_name):
        """Find file in directory that matches the base tile name"""
        possible_patterns = [
            f"{base_tile_name}.tif",
            f"{base_tile_name}_*.tif",  # For files with suffixes like _2020
        ]
        
        for pattern in possible_patterns:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]  # Return first match
        
        return None

    def visualize(self, n_examples=5):
        # Get output files as reference
        print('start visualization')
        output_files = list(self.output_dir.glob("*.tif"))
        
        if n_examples and n_examples < len(output_files):
            output_files = sorted(random.sample(output_files, n_examples))
        else:
            output_files = sorted(output_files)

        for output_path in output_files:
            tile = output_path.stem
            base_tile = self.extract_base_name(tile)
            
            # Find matching input files for all modalities
            input_data = {}
            missing_inputs = []
            
            for modality in self.input_modalities:
                matching_file = self.find_matching_file(self.input_dirs[modality], base_tile)
                if matching_file is None:
                    missing_inputs.append(modality)
                else:
                    input_data[modality] = matching_file
            
            if missing_inputs:
                print(f"Skipping {tile}: missing {missing_inputs}")
                continue

            # Load output data
            output_arr = rxr.open_rasterio(output_path).squeeze().values
            if output_arr.ndim == 2:
                output_arr = output_arr[None, ...]
            output_tensor = torch.tensor(output_arr).float().unsqueeze(0)
            output_vis = render_output(self.base_output_modality, output_tensor)

            # Create visualization
            n_inputs = len(self.input_modalities)
            fig, axes = plt.subplots(1, n_inputs + 1, figsize=((n_inputs + 1) * 4, 6))
            
            # Handle single subplot case
            if n_inputs + 1 == 1:
                axes = [axes]

            # Plot each input modality WITH CROPPING
            for i, (modality, input_path) in enumerate(input_data.items()):
                input_arr = rxr.open_rasterio(input_path).squeeze().values
                if input_arr.ndim == 2:
                    input_arr = input_arr[None, ...]
                
                input_tensor = torch.tensor(input_arr).float().unsqueeze(0)
                
                # Apply the same crop as in generation
                crop = T.CenterCrop(self.crop_size)
                input_tensor = crop(input_tensor)
                
                input_vis = render_output(modality, input_tensor)
                
                # Process visualization tensor
                if input_vis.ndim == 4:
                    input_vis = input_vis.squeeze(0)
                if input_vis.ndim == 3 and input_vis.shape[0] in [1, 3]:
                    input_vis = input_vis.permute(1, 2, 0)
                
                axes[i].imshow(input_vis.cpu().numpy())
                axes[i].set_title(f"{modality} Input ({self.crop_size}x{self.crop_size})")
                axes[i].axis("off")

            # Plot output
            if output_vis.ndim == 4:
                output_vis = output_vis.squeeze(0)
            if output_vis.ndim == 3 and output_vis.shape[0] in [1, 3]:
                output_vis = output_vis.permute(1, 2, 0)

            axes[n_inputs].imshow(output_vis.cpu().numpy())
            axes[n_inputs].set_title(f"Generated {self.base_output_modality}")
            axes[n_inputs].axis("off")

            plt.suptitle(f"{self.input_modality} → {self.output_modality}: {tile}")
            plt.tight_layout()

            vis_path = self.vis_dir / f"{tile}.png"
            plt.savefig(vis_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"Saved comparison: {vis_path}")

            # Print LULC stats if applicable (using cropped data)
            for modality, input_path in input_data.items():
                if modality == "LULC":
                    input_arr = rxr.open_rasterio(input_path).squeeze().values
                    if input_arr.ndim == 2:
                        input_arr = input_arr[None, ...]
                    cropped_tensor = crop(torch.tensor(input_arr).unsqueeze(0))
                    print_lulc_stats("Input (cropped)", cropped_tensor.squeeze().numpy())
            
            if self.base_output_modality == "LULC":
                print_lulc_stats("Output", output_arr.squeeze())