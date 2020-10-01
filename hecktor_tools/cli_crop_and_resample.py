"""
Script for resampling and cropping the images
Adapted from github.com/voreille/hecktor

        This command line interface allows to resample NIFTI files within a
        given bounding box contain in BOUNDING_BOXES_FILE. The images are
        resampled with spline interpolation
        of degree --order (default=3) and the segmentation are resampled
        by nearest neighbor interpolation.
        INPUT_FOLDER is the path of the folder containing the NIFTI to
        resample.
        OUTPUT_FOLDER is the path of the folder where to store the
        resampled NIFTI files.
        BOUNDING_BOXES_FILE is the path of the .csv file containing the
        bounding boxes of each patient.
    
"""

import os
from multiprocessing import Pool
import glob
import argparse
import logging

import pandas as pd
import numpy as np
from scipy.ndimage import affine_transform
from scipy.interpolate import RegularGridInterpolator
import SimpleITK as sitk


from resampling import Resampler, get_sitk_volume_from_np, get_np_volume_from_sitk

# Constants
DEFAULT_SOURCE_DIR = "../../../Datasets/HECKTOR/hecktor_train/hecktor_nii"
DEFAULT_TARGET_DIR = "../../../Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
DEFAULT_BB_FILEPATH = "./outputs/bboxes_FH.csv"
DEFAULT_NEW_SPACING = [1.0, 1.0, 3.0]  # New spacing to resample to in mm -- (W,H,D) format
DEFAULT_CORES = 24
DEFAULT_ORDER = 3


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
                        help="Directory containing patient folders"
                        )

    parser.add_argument("--bbox_filepath", 
                        type=str, 
                        default=DEFAULT_BB_FILEPATH,
                        help="CSV file that contains bbox coordinates"
                        )

    parser.add_argument("--new_spacing", 
                        type=float, 
                        nargs=3,
                        default=DEFAULT_NEW_SPACING,
                        help="Voxel spacing the output images in mm -- (W,H,D) format"
                        )

    parser.add_argument("--cores", 
                        type=int, 
                        default=DEFAULT_CORES,
                        help="Number of workers for parallelization"
                        )

    parser.add_argument("--order", 
                        type=int, 
                        default=DEFAULT_ORDER,
                        help="Order of the spline interpolation used to resample"
                        )


    args = parser.parse_args()
    return args


def main(args):

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)
    print("Resampling to spacing:", args.new_spacing)
    bb_df = pd.read_csv(args.bbox_filepath)
    bb_df = bb_df.set_index('PatientID')
    files_list = [
        f for f in glob.glob(args.source_dir + '/**/*.nii.gz', recursive=True)
    ]
    resampler = Resampler(bb_df, args.target_dir, args.order, resampling=args.new_spacing)
    with Pool(args.cores) as p:
        p.map(resampler, files_list)



if __name__ == '__main__':
    args = get_args()
    main(args)