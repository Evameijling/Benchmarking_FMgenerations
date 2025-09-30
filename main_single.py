from src.generations import TerraMindGenerator
from src.visualize import ComparisonVisualizer, Visualizer
from src.utils import BandStacker
from pathlib import Path
import random
import numpy as np

ROOT = Path("/home/egm/Data/Projects/CopGen") 

if __name__ == "__main__":
    # TO PRINT GROUND TRUTH

    crop_size = 256  # Define crop size here

    InOutput = "output"
    modality = "S1RTC_from_S2L2A"
    tile = "433U_63L"
    visualizer = Visualizer(
        InOutput=InOutput,
        modality=modality,
        tile=tile,
        root=ROOT,
        crop_size=crop_size  # Pass crop_size to single visualizer
    )
    visualizer.visualize(save=True, show=False)