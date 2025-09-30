from src.generations import TerraMindGenerator
from src.visualize import ComparisonVisualizer, Visualizer
from src.utils import BandStacker
from pathlib import Path
import random
import numpy as np

ROOT = Path("/home/egm/Data/Projects/CopGen") 

if __name__ == "__main__":
    # SEED = 42
    # random.seed(SEED)
    # np.random.seed(SEED)
    # print(f"Set random seed to {SEED} for reproducible file selection")

    input_modalities = ["S1RTC", "LULC"]  # [DEM, LULC, S1RTC, S2L2A]
    output_modality = "S2L2A" # [DEM, LULC, S1RTC, S2L2A]
    max_files = 10
    crop_size = 256  # Define crop size here

    # Step 0: Stack bands for all input modalities
    for modality in input_modalities:
        dir_path = ROOT / "data" / "input" / modality
        if not dir_path.exists() or not any(dir_path.iterdir()):
            print(f"Stacking bands for {modality} (max {max_files} files)...")
            BandStacker(modality=modality, root=ROOT).stack_all(max_files=max_files)
        else:
            print(f"Bands for {modality} already stacked.")

    # Step 1: Generate outputs
    output_dir = ROOT / "data" / "output" / f"{output_modality}_from_{'_'.join(input_modalities)}"
    if not output_dir.exists() or not any(output_dir.iterdir()):
        print(f"Generating {output_modality} from {input_modalities} (max {max_files} files)...")
        generator = TerraMindGenerator(
            input_modalities=input_modalities,
            output_modality=output_modality,
            model_name="terramind_v1_base_generate",
            crop_size=crop_size,  # Use the defined crop_size
            timesteps=10,
            pretrained=True,
            standardize=True,
            device="cuda",
            root=ROOT   
        )
        generator.process_all(max_files=max_files)
    else:
        print(f"Outputs for {output_modality} already generated.")

    # Step 2: Visualize comparisons
    visualizer = ComparisonVisualizer(
        input_modality=input_modalities,  
        output_modality=output_modality,
        root=ROOT,
        crop_size=crop_size  # Pass crop_size to visualizer
    )
    visualizer.visualize(n_examples=10)

    # # Visualize a single file and save
    # InOutput = "output"
    # modality = "S1RTC_from_S2L2A"
    # tile = "433U_63L"
    # visualizer = Visualizer(
    #     InOutput=InOutput,
    #     modality=modality,
    #     tile=tile,
    #     root=ROOT
    # )
    # visualizer.visualize(save=True, show=False)