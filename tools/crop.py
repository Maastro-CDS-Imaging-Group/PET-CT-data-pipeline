"""
Performs cropping all images from a given directory and stores them in a new directory 

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



# TODO Modify main()

def main(args):

    new_spacing = [float(s) for s in args.new_spacing]
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    patient_ids = [str(dir_path).split('/')[-1] \
                   for dir_path in sorted(list(source_dir.glob("*/")))]

    print("Total patients found:", len(patient_ids))

    for p_id in tqdm(patient_ids):
        source_patient_dir = source_dir / Path(p_id)

        # CT
        ct_sitk = sitk.ReadImage(f"{source_patient_dir}/{p_id}_ct.nii.gz")
        ct_resampled_sitk = resample_sitk_image(ct_sitk, 
                                                new_spacing, 
                                                sitk_interpolator=SITK_INTERPOLATOR_DICT['linear'], 
                                                default_fill_value=DEFAULT_IMG_VOXEL_VALUE)
        
        # PET
        pet_sitk = sitk.ReadImage(f"{source_patient_dir}/{p_id}_pt.nii.gz")
        pet_resampled_sitk = resample_sitk_image(pet_sitk, 
                                                 new_spacing, 
                                                 sitk_interpolator=SITK_INTERPOLATOR_DICT['linear'], 
                                                 default_fill_value=DEFAULT_IMG_VOXEL_VALUE)

        # GTV mask
        gtv_sitk = sitk.ReadImage(f"{source_patient_dir}/{p_id}_ct_gtvt.nii.gz")
        gtv_resampled_sitk = resample_sitk_image(gtv_sitk, 
                                                 new_spacing, 
                                                 sitk_interpolator=SITK_INTERPOLATOR_DICT['nearest'], 
                                                 default_fill_value=DEFAULT_GTV_VOXEL_VALUE)

        # Write into target directory
        target_patient_dir = Path(f"{target_dir}/{p_id}")
        target_patient_dir.mkdir(parents=True, exist_ok=True)

        ct_resampled_path = target_patient_dir / Path(f"{p_id}_ct.nrrd")
        sitk.WriteImage(ct_resampled_sitk, str(ct_resampled_path), useCompression=True)

        pet_resampled_path = target_patient_dir / Path(f"{p_id}_pt.nrrd")
        sitk.WriteImage(pet_resampled_sitk, str(pet_resampled_path), useCompression=True)
        
        gtv_resampled_path = target_patient_dir / Path(f"{p_id}_ct_gtvt.nrrd")
        sitk.WriteImage(gtv_resampled_sitk, str(gtv_resampled_path), useCompression=True)
        


# ------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    main(args)