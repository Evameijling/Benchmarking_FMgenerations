import os
from pathlib import Path
import torch
import rioxarray as rxr
import numpy as np
from terratorch.registry import FULL_MODEL_REGISTRY
import torchvision.transforms as T
import rasterio
from tqdm import tqdm

class TerraMindGenerator:
    def __init__(
        self,
        input_modalities=["S1RTC"],
        output_modality="S2L2A",
        root=None,
        crop_size=(256, 256),
        device=None,
        model_name="terramind_v1_base_generate",
        timesteps=50,
        standardize=True,
        pretrained=True,
    ):
        self.root = Path(root)
        self.input_modalities = input_modalities
        self.output_modality = output_modality
        self.input_dirs = {mod: self.root / "data" / "input" / mod for mod in input_modalities}
        self.output_dir = self.root / "data" / "output" / f'{output_modality}_from_{"_".join(input_modalities)}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_size = crop_size
        self.timesteps = timesteps

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        print(f"Using device: {self.device}")

        self.model = FULL_MODEL_REGISTRY.build(
            model_name,
            modalities=self.input_modalities,
            output_modalities=[self.output_modality],
            pretrained=pretrained,
            standardize=standardize,
        )
        self.model = self.model.to(self.device).eval()

    def extract_base_name(self, filename):
        """Extract base tile name, removing year suffixes"""
        base = filename
        if '_20' in base:  # Remove _2020, _2021, etc.
            base = base.split('_20')[0]
        return base

    def find_matching_file(self, modality, base_tile_name):
        """Find file in modality directory that matches the base tile name"""
        possible_patterns = [
            f"{base_tile_name}.tif",
            f"{base_tile_name}_*.tif",  # For files with suffixes like _2020
        ]
        
        for pattern in possible_patterns:
            matches = list(self.input_dirs[modality].glob(pattern))
            if matches:
                return matches[0]  # Return first match
        
        raise FileNotFoundError(f"No matching file found for {base_tile_name} in {modality}")

    def load_and_preprocess(self, modality, reference_tif_path):
        # Extract base name from reference file
        base_name = self.extract_base_name(reference_tif_path.stem)
        
        # Find matching file in this modality
        actual_file = self.find_matching_file(modality, base_name)
        print(f"Loading {modality}: {actual_file.name} (base: {base_name})")
        
        arr = rxr.open_rasterio(actual_file).squeeze().values.astype(np.float32)

        # ... rest of existing preprocessing logic stays the same ...
        if modality == "S1RTC":
            epsilon = 1e-8
            arr = 10.0 * np.log10(np.clip(arr, epsilon, None))
            arr = np.clip(arr, -50, 10)
        elif modality == "LULC":
            if arr.ndim == 2:
                unique_classes = sorted(np.unique(arr))
                normalized_arr = np.zeros_like(arr, dtype=np.float32)
                for i, cls in enumerate(unique_classes[:10]):
                    normalized_arr[arr == cls] = float(i)
                arr = normalized_arr[None, ...]
                temp_tensor = torch.tensor(arr).float().unsqueeze(0).to(self.device)
                if hasattr(self.model, 'tokenizer') and modality in self.model.tokenizer:
                    tokenizer = self.model.tokenizer[modality]
                    if hasattr(tokenizer, 'encoder') and hasattr(tokenizer.encoder, 'proj'):
                        expected_channels = tokenizer.encoder.proj.in_channels
                        if expected_channels == 10:
                            one_hot = torch.zeros(1, 10, arr.shape[1], arr.shape[2], device=self.device)
                            class_indices = temp_tensor.long()
                            one_hot.scatter_(1, class_indices, 1.0)
                            arr = one_hot.squeeze(0).cpu().numpy()
        elif modality in ["DEM"] and arr.ndim == 2:
            arr = arr[None, ...]

        return arr

    def process_all(self, max_files=None):
        # Sort for consistency
        sample_files = sorted(list(self.input_dirs[self.input_modalities[0]].glob("*.tif")))
        if max_files:
            sample_files = sample_files[:max_files]

        print(f"Found {len(sample_files)} files to process")
        print(f"First few files: {[f.name for f in sample_files[:3]]}")

        for tif_path in tqdm(sample_files, desc=f"Generating {self.output_modality} from {self.input_modalities}"):
            self.process_file(tif_path)

    def process_file(self, tif_path):
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        all_tensors = []
        for modality in self.input_modalities:
            arr = self.load_and_preprocess(modality, tif_path)
            tensor = torch.tensor(arr).float().unsqueeze(0).to(self.device)  # [1, C, H, W]
            crop = T.CenterCrop(self.crop_size)
            tensor = crop(tensor)
            all_tensors.append(tensor)

        input_dict = {mod: tensor for mod, tensor in zip(self.input_modalities, all_tensors)}

        with torch.no_grad():
            generated = self.model(input_dict, timesteps=self.timesteps, verbose=False)

        output = generated[self.output_modality].cpu().squeeze().numpy()

        tile_name = tif_path.stem
        out_path = self.output_dir / f"{tile_name}.tif"

        # Process output based on modality
        if self.output_modality == "S1RTC":
            # Convert dB back to linear power for output (to match input format and visualization)
            output = np.clip(output, -50, 10)
            output_data = (10 ** (output / 10)).astype(np.float32)
            output_dtype = np.float32
            
        elif self.output_modality == "S2L2A":
            # S2L2A: uint16 for reflectance values (0-10000 range)
            output_dtype = np.uint16
            output_data = np.clip(output, 0, 10000).astype(np.uint16)
            
        elif self.output_modality == "LULC":
            # LULC: discrete class indices as uint8
            print(f"LULC output shape: {output.shape}")
            print(f"LULC output stats - min: {output.min():.2f}, max: {output.max():.2f}, unique values: {len(np.unique(output))}")
            
            if np.allclose(output, np.round(output)):
                output_data = np.clip(output, 0, 255).astype(np.uint8)
            else:
                output_data = np.round(np.clip(output, 0, 255)).astype(np.uint8)
            
            output_dtype = np.uint8
            
            # Debug: Show class distribution
            unique_classes, counts = np.unique(output_data, return_counts=True)
            print(f"Generated LULC classes: {unique_classes[:10]}...")
            print(f"Most common class: {unique_classes[np.argmax(counts)]} ({counts.max()} pixels)")
            
        elif self.output_modality == "DEM":
            # DEM: float32 for elevation values
            output_dtype = np.float32
            output_data = output.astype(np.float32)
            
        else:
            # Default: float32
            output_dtype = np.float32
            output_data = output.astype(np.float32)

        with rasterio.open(tif_path) as src:
            meta = src.meta.copy()
            count = output_data.shape[0] if output_data.ndim == 3 else 1

        # Fix nodata based on dtype (from your old working code)
        if output_dtype == np.uint8:
            nodata_value = 255  # Use max value for uint8
        elif output_dtype == np.uint16:
            nodata_value = 65535  # Valid nodata for uint16
        elif output_dtype == np.float32:
            nodata_value = -9999.0  # Valid nodata for float
        else:
            nodata_value = None  # Remove if unsure
        
        meta.update({
            'count': count,
            'height': output_data.shape[-2],
            'width': output_data.shape[-1],
            'dtype': output_dtype,
        })
        
        # Only set nodata if we have a valid value
        if nodata_value is not None:
            meta['nodata'] = nodata_value
        else:
            meta.pop('nodata', None)
        
        with rasterio.open(out_path, 'w', **meta) as dst:
            if output_data.ndim == 2:
                dst.write(output_data, 1)
            else:
                dst.write(output_data)

if __name__ == "__main__":
    generator = TerraMindGenerator(
        input_modalities=["S1RTC", "DEM"],
        output_modality="S2L2A",
        root="/your/path/here"
    )
    generator.process_all()
