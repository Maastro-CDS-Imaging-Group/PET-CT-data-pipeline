"""

Plot HU and SUV distributions for CT and PET respectively.

Note: Distribution is relevant only *after* resampling all images to common voxel spacing

"""

import os, argparse
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

import SimpleITK as sitk


# Constants
DEFAULT_DATA_DIR = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFHN_rs113_hecktor_nii"
DEFAULT_PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
DEFAULT_OUTPUT_DIR = "../hecktor_meta/full_head_neck_crop"
DEFAULT_HAS_SUBDIRS = 0
DEFAULT_DATA_INFO = "crFHN_rs113_train"
DEFAULT_HU_WINDOW = None
DEFAULT_SUV_WINDOW = None
DEFAULT_PLOT_KDE = 0


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
                        help="HU window to apply to the CT images before plotting the distribution: [win_lo, win_hi]")

    parser.add_argument("--suv_window",
                        type=float,
                        nargs=2,
                        default=DEFAULT_SUV_WINDOW,
                        help="SUV window to apply to the PET images before plotting the distribution: [win_lo, win_hi]")

    parser.add_argument("--plot_kde",
                        type=int,
                        default=DEFAULT_PLOT_KDE,
                        help="0: No, 1: Yes")

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
    plot_kde = args.plot_kde == 1

    with open(patient_id_file, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split("\n") if p_id != ""]

    fig_ct, axs_ct = plt.subplots(2, 1, figsize=(10,10))
    fig_pet, axs_pet = plt.subplots(2, 1, figsize=(10,10))

    # To find the max and min values across all images
    HU_minmax_dict = {}
    SUV_minmax_dict = {}

    for p_id in tqdm(patient_ids):

        # For CT -------------------------------
        if has_subdirs:
            ct_img_path = f"{data_dir}/{p_id}/{p_id}_ct.nii.gz"
        else:
            ct_img_path = f"{data_dir}/{p_id}_ct.nii.gz"

        CT_sitk = sitk.ReadImage(ct_img_path)

        # HU histogram
        CT_np = sitk.GetArrayFromImage(CT_sitk)

        bin_size = 20
        if hu_window is None:
            CT_np = np.clip(CT_np, -1000, CT_np.max()) # Make non-valid region HU equal to air HU

            if plot_kde:
                kde = stats.gaussian_kde(CT_np.flatten())
                x_range = np.linspace(-1500, 2000, 3500)
                axs_ct[0].plot(x_range, kde(x_range))
            else:
                axs_ct[0].hist(CT_np.flatten(), bins=np.arange(-1500, 2000, bin_size), histtype='step')

        else:
            CT_np = np.clip(CT_np, hu_window[0], hu_window[1])
            x_axis_margin = 50
            axs_ct[0].hist(CT_np.flatten(), bins=np.arange(hu_window[0]-x_axis_margin, hu_window[1]+x_axis_margin, bin_size), histtype='step')

        # Min and max HU values for the patient
        HU_minmax_dict[p_id] = (CT_np.min(), CT_np.max())


        # For PET -------------------------------
        if has_subdirs:
            pet_img_path = f"{data_dir}/{p_id}/{p_id}_pt.nii.gz"
        else:
            pet_img_path = f"{data_dir}/{p_id}_pt.nii.gz"

        PET_sitk = sitk.ReadImage(pet_img_path)

        # SUV histogram
        PET_np = sitk.GetArrayFromImage(PET_sitk)

        bin_size = 0.25
        if suv_window is None:

            if plot_kde:
                kde = stats.gaussian_kde(PET_np.flatten())
                x_range = np.linspace(-1, 20, 1000)
                axs_pet[0].plot(x_range, kde(x_range))
            else:
                axs_pet[0].hist(PET_np.flatten(), bins=np.arange(-1, 30, bin_size), histtype='step')

        else:
            PET_np = np.clip(PET_np, suv_window[0], suv_window[1])
            x_axis_margin = 1
            axs_pet[0].hist(PET_np.flatten(), bins=np.arange(suv_window[0]-x_axis_margin, suv_window[1]+x_axis_margin, bin_size), histtype='step')

        # Min and max SUV values for the patient
        SUV_minmax_dict[p_id] = (PET_np.min(), PET_np.max())


    # Save histogram plots
    print("Saving the plots ...", end='')

    axs_ct[0].set_xlabel("HU")
    axs_ct[0].set_ylabel("Number of voxels")

    axs_pet[0].set_xlabel("SUV")
    axs_pet[0].set_ylabel("Number of voxels")

    # Plot and save min max line plots
    list_of_mins = [val[0] for val in HU_minmax_dict.values()]
    list_of_maxs = [val[1] for val in HU_minmax_dict.values()]
    axs_ct[1].plot(patient_ids, list_of_mins, 'r', label="Minimum HU per patient")
    axs_ct[1].plot(patient_ids, list_of_maxs, 'b', label="Maximum HU per patient")
    axs_ct[1].set_ylabel("HU")
    axs_ct[1].set_xticklabels(patient_ids, rotation=90)
    axs_ct[1].legend()

    list_of_mins = [val[0] for val in SUV_minmax_dict.values()]
    list_of_maxs = [val[1] for val in SUV_minmax_dict.values()]
    axs_pet[1].plot(patient_ids, list_of_mins, 'r', label="Minimum SUV per patient")
    axs_pet[1].plot(patient_ids, list_of_maxs, 'b', label="Maximum SUV per patient")
    axs_pet[1].set_ylabel("SUV")
    axs_pet[1].set_xticklabels(patient_ids, rotation=90)
    axs_pet[1].legend()

    # Save plots
    if hu_window is None:
        output_filepath = f"{output_dir}/{data_info}-CT_intensity_stats.png"
    else:
        output_filepath = f"{output_dir}/{data_info}-CT_intensity_stats_win.png"
    fig_ct.savefig(output_filepath)

    if suv_window is None:
        output_filepath = f"{output_dir}/{data_info}-PET_intensity_stats.png"
    else:
        output_filepath = f"{output_dir}/{data_info}-PET_intensity_stats_win.png"
    fig_pet.savefig(output_filepath)

    print("done")




if __name__ == '__main__':
    args = get_args()
    main(args)