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
        timesteps=50,
        standardize=True,
        pretrained=True,
    ):
        self.root = Path(root)
        self.input_modality = input_modality
        self.output_modality = output_modality
        self.input_dir = self.root / "data" / "input" / input_modality
        self.output_dir = self.root / "data" / "output" / f'{output_modality}_from_{input_modality}'
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
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        arr = rxr.open_rasterio(tif_path).squeeze().values  # shape: [C, H, W]
        arr = arr.astype(np.float32)
        if self.input_modality == "S1RTC":
            # Convert from linear power to dB safely
            epsilon = 1e-8
            arr = 10.0 * np.log10(np.clip(arr, epsilon, None))
            arr = np.clip(arr, -50, 10)  # Optional: match TerraMesh dB range ## CHECK! TODO
        elif self.input_modality == "S2L2A":
            # Convert from DN to reflectance
            # arr = arr / 1000.0 # Rescaling DN-->reflectance for S2L2A input (0-10 range, as expected by TerraMind)
            # arr = np.clip(arr, 0.0, 1.1)  # Optional: cap reflectance ## CHECK! TODO
            pass
        elif self.input_modality == "LULC":
            # LULC preprocessing - check if we need to expand dimensions
            print(f"LULC input stats - min: {arr.min()}, max: {arr.max()}, unique: {len(np.unique(arr))}")
            
            # Check if LULC tokenizer expects multi-channel input (like the 10-channel output we saw)
            if arr.ndim == 2:
                # Try expanding to match the expected LULC format
                # Based on the 10-channel output, the tokenizer might expect 10-channel input
                print("LULC input is single-channel, may need to expand dimensions")
                # For now, let's try with single channel and see the error details
                pass
            pass
        elif self.input_modality == "DEM":
            # DEM preprocessing if needed
            pass

        # Handle single-band files (expected for LULC and DEM)
        if arr.ndim == 2:
            if self.input_modality in ["LULC", "DEM"]:
                # Single-band is expected - add channel dimension
                arr = arr[None, ...]  # Add channel dimension: [H, W] -> [1, H, W]
                print(f"Added channel dimension: {arr.shape}")
            else:
                tqdm.write(f"WARNING: {tif_path} is not a multi-band file.")
                return

        img_tensor = torch.tensor(arr).float().unsqueeze(0).to(self.device)  # [1, C, H, W]
        print(f"Tensor shape before crop: {img_tensor.shape}")
        
        # Center crop
        crop = T.CenterCrop(self.crop_size)
        img_tensor_cropped = crop(img_tensor)
        
        print(f"Final tensor shape: {img_tensor_cropped.shape}")
        print(f"Tensor stats - min: {img_tensor_cropped.min():.3f}, max: {img_tensor_cropped.max():.3f}")

        # Inference
        with torch.no_grad():
            generated = self.model(img_tensor_cropped, timesteps=self.timesteps, verbose=False)

        output = generated[self.output_modality].cpu().squeeze().numpy()  # [C, H, W]

        # Output filename
        tile_name = tif_path.stem
        out_path = self.output_dir / f"{tile_name}.tif"

        # Choose dtype and process data based on output modality
        if self.output_modality == "S1RTC":
            # Convert dB back to linear power for output (to match input format and visualization)
            output = np.clip(output, -50, 10)  # Optional: enforce valid dB range before converting
            output_data = (10 ** (output / 10)).astype(np.float32)
            output_dtype = np.float32
        elif self.output_modality == "S2L2A":
            # S2L2A: uint16 for reflectance values (0-10000 range)
            output_dtype = np.uint16
            
            # Debug: Check the actual output range
            # print(f"Raw S2L2A output - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
            
            # The model outputs values in ~1000-5000 range, which is already reasonable for S2L2A
            # Just convert to uint16 without additional scaling
            output_data = np.clip(output, 0, 10000).astype(np.uint16)
                
            # print(f"Processed S2L2A output - min: {output_data.min()}, max: {output_data.max()}, mean: {output_data.mean():.1f}")
        
        elif self.output_modality == "LULC":
            # LULC: FSQ-VAE tokenized output - already discrete tokens/class indices
            # The model decoder reconstructs discrete token codes via tokenizer decode logic
            # No need to round - output should already be quantized class indices
            
            print(f"LULC output shape: {output.shape}")
            print(f"LULC output stats - min: {output.min():.2f}, max: {output.max():.2f}, unique values: {len(np.unique(output))}")
            
            # Check if output is already discrete (integer-like values)
            if np.allclose(output, np.round(output)):
                # Output is already discrete - just convert to appropriate integer type
                output_data = np.clip(output, 0, 255).astype(np.uint8)
            else:
                # If somehow continuous, we might need different handling
                # Could try nearest neighbor to closest valid token/class
                output_data = np.round(np.clip(output, 0, 255)).astype(np.uint8)
            
            output_dtype = np.uint8
            
            # Debug: Show class distribution
            unique_classes, counts = np.unique(output_data, return_counts=True)
            print(f"Generated LULC classes: {unique_classes[:10]}...")  # Show first 10
            print(f"Most common class: {unique_classes[np.argmax(counts)]} ({counts.max()} pixels)")
            print(f"LULC output_data shape after processing: {output_data.shape}")
            
        elif self.output_modality == "DEM":
            # DEM: typically float32 for elevation values
            output_dtype = np.float32
            output_data = output.astype(np.float32)
        else:
            # Default: keep as float32
            output_dtype = np.float32
            output_data = output.astype(np.float32)

        # Save as GeoTIFF using input metadata with specified dtype
        with rasterio.open(tif_path) as src:
            meta = src.meta.copy()
            
            # Handle single-band output (DEM) vs multi-band (S1RTC, S2L2A, LULC)
            if self.output_modality == "DEM":
                # Single-band output
                if output_data.ndim == 3 and output_data.shape[0] == 1:
                    output_data = output_data.squeeze(0)  # Remove channel dimension
                count = 1
            else:
                # Multi-band output (S1RTC, S2L2A, and LULC which has 10 bands)
                count = output_data.shape[0] if output_data.ndim == 3 else 1
                print(f"Setting count to {count} for {self.output_modality}")
            
            # Fix nodata based on dtype
            if output_dtype == np.uint8:
                meta['nodata'] = 255  # Use max value for uint8
            elif output_dtype == np.uint16:
                meta['nodata'] = 65535  # Valid nodata for uint16
            elif output_dtype == np.float32:
                meta['nodata'] = -9999.0  # Valid nodata for float
            else:
                meta.pop('nodata', None)  # Remove if unsure
                
            meta.update({
                'count': count, 
                'height': output_data.shape[-2], 
                'width': output_data.shape[-1],
                'dtype': output_dtype
            })

            with rasterio.open(out_path, 'w', **meta) as dst:
                if output_data.ndim == 2:
                    dst.write(output_data, 1)  # Single band
                else:
                    dst.write(output_data)  # Multi-band

if __name__ == "__main__":
    generator = TerraMindGenerator(
        input_modality="S1RTC",
        output_modality="S2L2A",
    )
    generator.process_all()