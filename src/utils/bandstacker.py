from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import numpy as np
from tqdm import tqdm

class BandStacker:
    def __init__(self, modality, root=None, target_resolution=10):
        self.modality = modality
        self.target_resolution = target_resolution
        self.root = Path(root) if root else Path(__file__).parent.parent.parent.resolve()
        self.input_root = self.root / "data" / "cop-gen-small-test" / f"Core-{modality}"
        self.output_root = self.root / "data" / "input" / modality
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.modality not in ["DEM", "LULC", "S1RTC", "S2L2A"]:
            raise ValueError("mode must be 'DEM', 'LULC', 'S1RTC', or 'S2L2A'")

    def _resample_to_target_resolution(self, src_path, target_transform, target_width, target_height):
        """Resample a raster to target resolution using nearest neighbor interpolation"""
        with rasterio.open(src_path) as src:
            data = src.read(
                out_shape=(src.count, target_height, target_width),
                resampling=Resampling.nearest
            )
            return data[0] if src.count == 1 else data

    def _get_target_resolution_params(self, reference_path):
        """Get target transform and dimensions for 10m resolution"""
        with rasterio.open(reference_path) as src:
            original_transform = src.transform
            pixel_size_x = abs(original_transform[0])
            pixel_size_y = abs(original_transform[4])
            
            # Scale factor to reach target resolution
            scale_x = pixel_size_x / self.target_resolution
            scale_y = pixel_size_y / self.target_resolution
            
            # New dimensions
            new_width = int(src.width * scale_x)
            new_height = int(src.height * scale_y)
            
            # New transform
            new_transform = rasterio.transform.from_bounds(
                *src.bounds, new_width, new_height
            )
            
            return new_transform, new_width, new_height

    def stack_all(self, max_files=None):
        """Stack all bands for the specified modality"""
        if self.modality == "S1RTC":
            self._stack_s1(max_files)
        elif self.modality == "S2L2A":
            self._stack_s2(max_files)
        elif self.modality == "LULC":
            self._process_lulc(max_files)

    def _stack_s1(self, max_files=None):
        # S1RTC doesn't need resampling - just stack VV and VH
        vv_paths = list(self.input_root.rglob("vv.tif"))
        if max_files is not None:
            vv_paths = vv_paths[:max_files]
            
        for vv_path in tqdm(vv_paths):
            vh_path = vv_path.parent / "vh.tif"
            if not vh_path.exists():
                continue  # skip if VH missing

            with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                vv = vv_src.read(1)
                vh = vh_src.read(1)
                meta = vv_src.meta.copy()
                meta.update(count=2)
                stack = np.stack([vv, vh])

            tile_name = vv_path.parent.parent.name
            output_path = self.output_root / f"{tile_name}.tif"
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(stack)
                dst.update_tags(SOURCE=str(vv_path.parent.relative_to(self.input_root.parent.parent)))

    def _stack_s2(self, max_files=None):
        # S2L2A bands need resampling to 10m resolution
        band_order = [
            "B01.tif", "B02.tif", "B03.tif", "B04.tif", "B05.tif", "B06.tif", "B07.tif",
            "B08.tif", "B8A.tif", "B09.tif", "B11.tif", "B12.tif"
        ]
        
        print(f"Looking for S2L2A data in: {self.input_root}")
        
        folders = list(self.input_root.rglob("*"))
        folders = [f for f in folders if f.is_dir()]
        
        print(f"Found {len(folders)} directories total")
        
        # Filter for directories that actually contain band files
        valid_folders = []
        for folder in folders:
            band_paths = [folder / b for b in band_order]
            if all(p.exists() for p in band_paths):
                valid_folders.append(folder)
            else:
                missing_bands = [b for b in band_order if not (folder / b).exists()]
                if len(missing_bands) < len(band_order):  # Some bands exist
                    print(f"Folder {folder.name} missing bands: {missing_bands[:3]}...")
        
        print(f"Found {len(valid_folders)} valid S2L2A folders with all bands")
        
        if max_files is not None:
            valid_folders = valid_folders[:max_files]
            print(f"Processing first {len(valid_folders)} folders")
        
        if not valid_folders:
            print("No valid S2L2A folders found! Check your data structure.")
            print("Expected structure: Core-S2L2A/TILE_ID/B01.tif, B02.tif, etc.")
            return
        
        for folder in tqdm(valid_folders):
            band_paths = [folder / b for b in band_order]
            
            # Use the first available 10m band (B02, B03, B04, or B08) as reference for target resolution
            reference_bands = ["B02.tif", "B03.tif", "B04.tif", "B08.tif"]
            reference_path = None
            for ref_band in reference_bands:
                ref_path = folder / ref_band
                if ref_path.exists():
                    reference_path = ref_path
                    break
            
            if reference_path is None:
                print(f"No reference band found in {folder}")
                continue  # skip if no reference band found
            
            # Get target resolution parameters
            target_transform, target_width, target_height = self._get_target_resolution_params(reference_path)
            
            bands = []
            meta = None
            for band_path in band_paths:
                # Resample each band to target resolution
                band = self._resample_to_target_resolution(band_path, target_transform, target_width, target_height)
                bands.append(band)
                
                if meta is None:
                    with rasterio.open(band_path) as src:
                        meta = src.meta.copy()
            
            # Update metadata for stacked output
            meta.update({
                'count': 12,
                'transform': target_transform,
                'width': target_width,
                'height': target_height
            })
            
            stack = np.stack(bands)
            tile_name = folder.parent.name if folder.parent.name != f"Core-{self.modality}" else folder.name
            output_path = self.output_root / f"{tile_name}.tif"
            
            # print(f"Saving stacked S2L2A to: {output_path}")
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(stack)
                dst.update_tags(SOURCE=str(folder.relative_to(self.input_root.parent.parent)))
    
    def _process_lulc(self, max_files=None):
        """Process LULC files - copy to input folder with optional resampling"""
        print(f"Looking for LULC data in: {self.input_root}")
        
        # Look for LULC files 
        lulc_patterns = ["*.tif"]
        lulc_paths = []
        
        for pattern in lulc_patterns:
            lulc_paths.extend(list(self.input_root.rglob(pattern)))
        
        # Remove duplicates and filter for actual files
        lulc_paths = list(set([p for p in lulc_paths if p.is_file()]))
        
        print(f"Found {len(lulc_paths)} LULC files")
        
        if max_files is not None:
            lulc_paths = lulc_paths[:max_files]
            print(f"Processing first {len(lulc_paths)} files")
        
        if not lulc_paths:
            print("No LULC files found! Check your data structure.")
            print("Expected structure: Core-LULC/TILE_ID/*.tif")
            return
        
        for lulc_path in tqdm(lulc_paths):
            with rasterio.open(lulc_path) as src:
                # Check if resampling is needed
                pixel_size_x = abs(src.transform[0])
                pixel_size_y = abs(src.transform[4])
                current_resolution = min(pixel_size_x, pixel_size_y)
                
                if abs(current_resolution - self.target_resolution) > 0.1:  # Need resampling
                    target_transform, target_width, target_height = self._get_target_resolution_params(lulc_path)
                    data = self._resample_to_target_resolution(lulc_path, target_transform, target_width, target_height)
                    
                    meta = src.meta.copy()
                    meta.update({
                        'transform': target_transform,
                        'width': target_width,
                        'height': target_height
                    })
                else:  # No resampling needed
                    data = src.read(1)
                    meta = src.meta.copy()
            
            # Determine output filename based on folder structure
            if lulc_path.parent.name == f"Core-{self.modality}":
                # File is directly in Core-LULC folder
                tile_name = lulc_path.stem
            else:
                # File is in a subfolder
                tile_name = lulc_path.parent.name
            
            output_path = self.output_root / lulc_path.name
            
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data, 1)
                dst.update_tags(SOURCE=str(lulc_path.relative_to(self.input_root.parent.parent)))
        
if __name__ == "__main__":
    BandStacker(modality="S1RTC").stack_all()