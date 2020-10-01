"""

Performs cropping on all images from a given directory and stores them in a new directory 

    - Crops away slices based on the following rules:
        1. Overhead space -- For a patient, use PET to find if there's background space over the patients head. 
                            If so, crop those overhead slices away from both PET and CT.
        2. Too many slices / Different slice numbers for PET and CT -- For a patient, crop both PET and CT images 
                                                                       to have a given physical axial size.   

    - Crops the width and height based on following rules:
        1. Different x-y sizes for PET and CT / Too much background space -- If the physical width or height is 
                                                                            greater than a specified value, 
                                                                            crop the extra parts away measuring 
                                                                            from the x-y plane's centre. 

TODO Finish the script, if needed. Or delete it
"""

import os
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

import multiprocessing as mp

import SimpleITK as sitk

# ------------------------------------------------
# Constants

DEFAULT_SOURCE_DIR = "../../../Datasets/HECKTOR/hecktor_train/rs113_hecktor_nii"
DEFAULT_TARGET_DIR = "../../../Datasets/HECKTOR/hecktor_train/rs113crop_hecktor_nii"

DEFAULT_SPACING = [1.0, 1.0, 3.0] # (W,H,D) (mm)

DEFAULT_PHY_DEPTH = 270.0  # (mm)
DEFAULT_PHY_WIDTH = 449.0  # (mm)
DEFAULT_PHY_HEIGHT = 449.0 # (mm)

# ------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", 
                        type=str, 
                        default=DEFAULT_SOURCE_DIR,
                        help="Directory containing patient folders"
                        )

    parser.add_argument("--target_dir", 
                        type=str, 
                        default=DEFAULT_TARGET_DIR,
                        help="Directory that will contain patient folders"
                        )

    parser.add_argument("--spacing", 
                        type=float,
                        nargs=3,
                        default=DEFAULT_SPACING,
                        help="Common spacing for all input images in mm -- (W,H,D) format"
                        )

    parser.add_argument("--output_phy_size", 
                        type=float, 
                        nargs=3,
                        default=[DEFAULT_PHY_WIDTH, DEFAULT_PHY_HEIGHT, DEFAULT_PHY_DEPTH],
                        help="Physical size of the output images in mm -- (W,H,D) format"
                        )

    args = parser.parse_args()
    return args

