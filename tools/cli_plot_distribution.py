"""

Plot HU and SUV distributions for CT and PET respectively. 

Note: Distribution is relevant only *after* resampling all images to common voxel spacing

"""

import os, argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import SimpleITK as sitk


# Constants
DEFAULT_DATA_DIR = "../../../Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
DEFAULT_PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
DEFAULT_OUTPUT_DIR = "../hecktor_meta"
DEFAULT_HAS_SUBDIRS = 0
DEFAULT_DATA_INFO = "train_crFH_rs113"
DEFAULT_HU_WINDOW = None
DEFAULT_SUV_WINDOW = None


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", 
                        type=str, 
                        default=DEFAULT_DATA_DIR,
                        help="Directory containing patient folders"
                        )
    
    parser.add_argument("--patient_id_file", 
                        type=str, 
                        default=DEFAULT_PATIENT_ID_FILE,
                        help="File containing patient IDs"
                        )

    parser.add_argument("--output_dir", 
                        type=str, 
                        default=DEFAULT_OUTPUT_DIR,
                        help="Directory to store the histogram plots in"
                        )

    parser.add_argument("--has_subdirs", 
                        type=int,
                        default=DEFAULT_HAS_SUBDIRS,
                        help="1, if patient dirs exist. 0, otherwise."
                        )
    
    parser.add_argument("--data_info", 
                        type=str,
                        default=DEFAULT_DATA_INFO,
                        help="Options: train_crFH_rs113, test_crFH_rs113, etc."
                        )

    parser.add_argument("--hu_window",
                        type=float,
                        nargs=2, 
                        default=DEFAULT_HU_WINDOW, 
                        help="HU window to apply to the CT images before plotting the distribution")

    parser.add_argument("--suv_window",
                        type=float, 
                        nargs=2, 
                        default=DEFAULT_SUV_WINDOW, 
                        help="SUV window to apply to the PET images before plotting the distribution")

    args = parser.parse_args()
    return args


def main(args):

    data_dir = args.data_dir
    patient_id_file = args.patient_id_file
    data_info = args.data_info
    output_dir = args.output_dir
    has_subdirs = args.has_subdirs == 1
    hu_window = args.hu_window
    suv_window = args.suv_window

    with open(patient_id_file, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split("\n") if p_id != ""]

    fig_ct, ax_ct = plt.subplots(figsize=(20,10))
    fig_pet, ax_pet = plt.subplots(figsize=(20,10))

    # To find the max and min values across all images
    HU_data_min, HU_data_max = 999, -999
    SUV_data_min, SUV_data_max = 999, -999

    for p_id in tqdm(patient_ids):

        # For CT -------------------------------
        if has_subdirs:
            ct_img_path = f"{data_dir}/{p_id}/{p_id}_ct.nii.gz"
        else:
            ct_img_path = f"{data_dir}/{p_id}_ct.nii.gz"

        ct_sitk = sitk.ReadImage(ct_img_path)
        
        # HU histogram
        ct_np = sitk.GetArrayFromImage(ct_sitk)
        
        bin_size = 10
        if hu_window is None:
            ax_ct.hist(ct_np.flatten(), bins=np.arange(-3020, 1700, bin_size), histtype='step')
        else:
            ct_np[ct_np < hu_window[0]] = hu_window[0]
            ct_np[ct_np > hu_window[1]] = hu_window[1]
            x_axis_margin = 50
            ax_ct.hist(ct_np.flatten(), bins=np.arange(hu_window[0]-x_axis_margin, hu_window[1]+x_axis_margin, bin_size), histtype='step')

        min_hu, max_hu = ct_np.min(), ct_np.max()
        if min_hu < HU_data_min:  HU_data_min = min_hu
        if max_hu > HU_data_max:  HU_data_max = max_hu
                

        # For PET -------------------------------
        if has_subdirs:
            pet_img_path = f"{data_dir}/{p_id}/{p_id}_pt.nii.gz"
        else:
            pet_img_path = f"{data_dir}/{p_id}_pt.nii.gz"

        pet_sitk = sitk.ReadImage(pet_img_path)

        # SUV histogram
        pet_np = sitk.GetArrayFromImage(pet_sitk)

        bin_size = 0.25
        if suv_window is None:
            ax_pet.hist(pet_np.flatten(), bins=np.arange(-1, 30, bin_size), histtype='step')
        else:
            pet_np[pet_np < suv_window[0]] = suv_window[0]
            pet_np[pet_np > suv_window[1]] = suv_window[1]
            x_axis_margin = 1
            ax_pet.hist(pet_np.flatten(), bins=np.arange(suv_window[0]-x_axis_margin, suv_window[1]+x_axis_margin, bin_size), histtype='step')

        min_suv, max_suv = pet_np.min(), pet_np.max()
        if min_suv < SUV_data_min:  SUV_data_min = min_suv
        if max_suv > SUV_data_max:  SUV_data_max = max_suv
        

    # Save histogram plots
    print("Saving the plots ...", end='')

    ax_ct.set_xlabel("HU")
    ax_ct.set_ylabel("Number of voxels")
    if hu_window is None:
        output_filepath = f"{output_dir}/hecktor_{data_info}_CT_hist.png"
    else:
        output_filepath = f"{output_dir}/hecktor_{data_info}_CT_win_hist.png"
    fig_ct.savefig(output_filepath)

    ax_pet.set_xlabel("SUV")
    ax_pet.set_ylabel("Number of voxels")
    if suv_window is None:
        output_filepath = f"{output_dir}/hecktor_{data_info}_PET_hist.png"
    else:
        output_filepath = f"{output_dir}/hecktor_{data_info}_PET_win_hist.png"
    fig_pet.savefig(output_filepath)

    print("done")

    print("Min HU found:", HU_data_min)
    print("Max HU found:", HU_data_max)
    print("Min SUV found:", SUV_data_min)
    print("Max SUV found:", SUV_data_max)


if __name__ == '__main__':
    args = get_args()
    main(args)