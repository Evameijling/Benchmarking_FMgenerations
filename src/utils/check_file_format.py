import rasterio
from pathlib import Path

def check_file_format(file_path):
    """Check and print the format details of a raster file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    try:
        with rasterio.open(file_path) as src:
            print(f"File: {file_path.name}")
            print(f"Data type: {src.dtypes[0]}")
            print(f"Shape: {src.shape} (Height x Width)")
            print(f"Bands: {src.count}")
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Bounds: {src.bounds}")
            if src.nodata is not None:
                print(f"NoData value: {src.nodata}")
            
            # Read a small sample to show value ranges
            sample = src.read(1, window=((0, min(100, src.height)), (0, min(100, src.width))))
            print(f"Sample values range: {sample.min()} to {sample.max()}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Add your file paths here
    files_to_check = [
        "/home/egm/Data/Projects/CopGen/data/input/S1RTC/example.tif",  # Replace with your actual path
        "/home/egm/Data/Projects/CopGen/data/input/S2L2A/example.tif",  # Replace with your actual path
    ]
    
    for file_path in files_to_check:
        check_file_format(file_path)