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
        input_modality="S1RTC",
        output_modality="S2L2A",
        root=None,
        crop_size=(256, 256),
        device=None,
        model_name="terramind_v1_base_generate",
        timesteps=10,
        standardize=True,
        pretrained=True,
    ):
        self.root = Path(root)
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.input_dir = self.root / "data" / "input" / input_modality
        self.output_dir = self.root / "data" / "output" / output_modality
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
            modalities=[self.input_modality],
            output_modalities=[self.output_modality],
            pretrained=pretrained,
            standardize=standardize,
        )
        self.model = self.model.to(self.device).eval()

    def process_all(self, max_files=None):
        """Process all files in the input directory"""
        # Get all .tif files
        tif_files = list(self.input_dir.glob("*.tif"))
        
        # Limit number of files if specified
        if max_files is not None:
            tif_files = tif_files[:max_files]
        
        print(f"Found {len(tif_files)} files to process")
        
        for tif_path in tqdm(tif_files, desc=f"Generating {self.output_modality} from {self.input_modality}"):
            self.process_file(tif_path)

    def process_file(self, tif_path):
        arr = rxr.open_rasterio(tif_path).squeeze().values  # shape: [C, H, W]
        if arr.ndim == 2:
            tqdm.write(f"WARNING: {tif_path} is not a multi-band file.")
            return
        img_tensor = torch.tensor(arr).float().unsqueeze(0).to(self.device)  # [1, C, H, W]

        # Center crop
        crop = T.CenterCrop(self.crop_size)
        img_tensor_cropped = crop(img_tensor)

        # Inference
        with torch.no_grad():
            generated = self.model(img_tensor_cropped, timesteps=self.timesteps, verbose=False)

        output = generated[self.output_modality].cpu().squeeze().numpy()  # [C, H, W]

        # Output filename
        tile_name = tif_path.stem
        out_path = self.output_dir / f"{tile_name}.tif"

        # Choose dtype and process data based on output modality
        if self.output_modality == "S1RTC":
            # S1RTC: float32 for backscatter values in dB (float16 not supported by rasterio)
            output_dtype = np.float32
            output_data = output.astype(np.float32)
        elif self.output_modality == "S2L2A":
            # S2L2A: uint16 for reflectance values (0-10000 range)
            output_dtype = np.uint16
            
            # Debug: Check the actual output range
            # print(f"Raw S2L2A output - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
            
            # The model outputs values in ~1000-5000 range, which is already reasonable for S2L2A
            # Just convert to uint16 without additional scaling
            output_data = np.clip(output, 0, 10000).astype(np.uint16)
                
            # print(f"Processed S2L2A output - min: {output_data.min()}, max: {output_data.max()}, mean: {output_data.mean():.1f}")
            
        else:
            # Default: keep as float32
            output_dtype = np.float32
            output_data = output.astype(np.float32)

        # Save as GeoTIFF using input metadata with specified dtype
        with rasterio.open(tif_path) as src:
            meta = src.meta.copy()
            meta.update({
                'count': output.shape[0], 
                'height': output.shape[1], 
                'width': output.shape[2],
                'dtype': output_dtype
            })
            
            # # Fix nodata value for different data types
            # if output_dtype == np.uint16:
            #     meta['nodata'] = 65535  # Use max value for uint16
            # elif output_dtype == np.float32:
            #     meta['nodata'] = -9999.0  # Standard float nodata
            
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(output_data)

if __name__ == "__main__":
    generator = TerraMindGenerator(
        input_modality="S1RTC",
        output_modality="S2L2A",
    )
    generator.process_all()