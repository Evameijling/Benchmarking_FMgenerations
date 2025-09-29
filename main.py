from src.generations import TerraMindGenerator
from src.visualize import ComparisonVisualizer, Visualizer
from src.utils import BandStacker
from pathlib import Path
import random
import numpy as np

ROOT = Path("/home/egm/Data/Projects/CopGen") 

if __name__ == "__main__":
    SEED = 42  # You can change this to any number you like
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Set random seed to {SEED} for reproducible file selection")

    input_modality = "S1RTC"   # [DEM, LULC, S1RTC, S2L2A]
    output_modality = "LULC"  # [DEM, LULC, S1RTC, S2L2A]
    max_files = 30  # Set maximum number of files to process (None for all)

    # Step 0: Stack bands if not already done 
    dir_path = ROOT / "data" / "input" / input_modality
    if not dir_path.exists() or not any(dir_path.iterdir()):
        print(f"Stacking bands for {input_modality} (max {max_files} files)...")
        BandStacker(modality=input_modality, root=ROOT).stack_all(max_files=max_files)
    else:
        print(f"Bands for {input_modality} already stacked.")

    # Step 1: Generate outputs (directories are set automatically)
    output_dir = ROOT / "data" / "output" / f"{output_modality}_from_{input_modality}"
    if not output_dir.exists() or not any(output_dir.iterdir()):
        print(f"Generating {output_modality} from {input_modality} (max {max_files} files)...")
        generator = TerraMindGenerator(
            input_modality=input_modality,
            output_modality=output_modality,
            model_name="terramind_v1_base_generate",
            crop_size=256,
            timesteps=10,
            pretrained=True,
            standardize=True,
            device="cuda",
            root=ROOT   
        )
        generator.process_all(max_files=max_files)
    else:
        print(f"Outputs for {output_modality} already generated.")

    # Step 2: Visualize comparisons (directories are set automatically)
    visualizer = ComparisonVisualizer(
        input_modality=input_modality,
        output_modality=output_modality,
        root=ROOT
    )
    visualizer.visualize(n_examples=15)  # Visualize 5 random examples
    
    # # Visualize a single file and save
    # InOutput = "output"  # [input, output]
    # modality = "LULC_from_S1RTC" # [DEM, LULC, S1RTC, S2L2A]
    # tile = "433U_1061L"
    # visualizer = Visualizer(
    #     InOutput=InOutput,  # or "output"
    #     modality=modality, 
    #     tile=tile,
    #     root=ROOT
    # )
    # visualizer.visualize(save=True, show=False)