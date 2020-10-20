"""

Obtain counts for voxel spacing and array sizes for all *original* CT and PET images from HECKTOR dataset.

"""

import os
import numpy as np
from tqdm import tqdm
import argparse

import SimpleITK as sitk


# Constants
DEFAULT_DATA_DIR = "../../../Datasets/HECKTOR/hecktor_train/crS_rs113_hecktor_nii"
DEFAULT_PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
DEFAULT_OUTPUT_FILE = "../hecktor_meta/default_small_crop/crS_rs113_train-spatial_stats.txt"
DEFAULT_HAS_SUBDIRS = 0
DEFAULT_DATA_INFO = "crS_rs113_train"


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

    parser.add_argument("--output_file", 
                        type=str, 
                        default=DEFAULT_OUTPUT_FILE,
                        help="File to store the results"
                        )

    parser.add_argument("--has_subdirs", 
                        type=int,
                        default=DEFAULT_HAS_SUBDIRS,
                        help="1, if patient dirs exist. 0, otherwise."
                        )

    parser.add_argument("--data_info", 
                        type=str,
                        default=DEFAULT_DATA_INFO,
                        help="Options: train, test, crFHN_rs113_train, crFHN_rs113_test, crS_rs113_train, crS_rs113_test."
                        )

    args = parser.parse_args()
    return args


def main(args):
	
	data_dir = args.data_dir
	patient_id_file = args.patient_id_file
	data_info = args.data_info
	has_subdirs = args.has_subdirs == 1

	with open(patient_id_file, 'r') as pf:
		patient_ids = [p_id for p_id in pf.read().split("\n") if p_id != ""]

	# patient_ids = sorted(os.listdir(data_dir))
	# print("Patients found:", len(patient_ids))

	ct_xy_spacing_counts = {}
	ct_z_spacing_counts = {}
	ct_xy_size_counts = {}
	ct_n_slices_counts = {}

	pet_xy_spacing_counts = {}
	pet_z_spacing_counts = {}
	pet_xy_size_counts = {}
	pet_n_slices_counts = {}

	
	for p_id in tqdm(patient_ids):

		# For CT -------------------------------
		if has_subdirs:
			ct_img_path = f"{data_dir}/{p_id}/{p_id}_ct.nii.gz"
		else:
			ct_img_path = f"{data_dir}/{p_id}_ct.nii.gz"

		ct_sitk = sitk.ReadImage(ct_img_path)

		# Spacing
		spacing = ct_sitk.GetSpacing() # (W,H,D)
		if round(spacing[0], 2) not in ct_xy_spacing_counts.keys(): # x-y spacing
			ct_xy_spacing_counts[round(spacing[0], 2)] = 1
		else:
			ct_xy_spacing_counts[round(spacing[0], 2)] += 1

		if round(spacing[2], 2) not in ct_z_spacing_counts.keys(): # z-spacing
			ct_z_spacing_counts[round(spacing[2], 2)] = 1
		else:
			ct_z_spacing_counts[round(spacing[2], 2)] += 1

		# Size
		size = ct_sitk.GetSize() # Size: (W,H,D)
		if size[0] not in ct_xy_size_counts.keys(): # x-y size
			ct_xy_size_counts[size[0]] = 1
		else:
			ct_xy_size_counts[size[0]] += 1

		if size[2] not in ct_n_slices_counts.keys(): # n-slices
			ct_n_slices_counts[size[2]] = 1
		else:
			ct_n_slices_counts[size[2]] += 1
		

		# For PET -------------------------------
		if has_subdirs:
			pet_img_path = f"{data_dir}/{p_id}/{p_id}_pt.nii.gz"
		else:
			pet_img_path = f"{data_dir}/{p_id}_pt.nii.gz"

		pet_sitk = sitk.ReadImage(pet_img_path)
		
		# Spacing
		spacing = pet_sitk.GetSpacing() # (W,H,D)
		if round(spacing[0], 2) not in pet_xy_spacing_counts.keys(): # x-y spacing
			pet_xy_spacing_counts[round(spacing[0], 2)] = 1
		else:
			pet_xy_spacing_counts[round(spacing[0], 2)] += 1

		if round(spacing[2], 2) not in pet_z_spacing_counts.keys(): # z-spacing
			pet_z_spacing_counts[round(spacing[2], 2)] = 1
		else:
			pet_z_spacing_counts[round(spacing[2], 2)] += 1

		# Size
		size = pet_sitk.GetSize() # (W,H,D)
		if size[0] not in pet_xy_size_counts.keys(): # x-y size
			pet_xy_size_counts[size[0]] = 1
		else:
			pet_xy_size_counts[size[0]] += 1
		
		if size[2] not in pet_n_slices_counts.keys(): # n-slices
			pet_n_slices_counts[size[2]] = 1
		else:
			pet_n_slices_counts[size[2]] += 1


	# Write results into file
	# output_stats_filepath = f"{output_dir}/hecktor_{dataset}_stats.txt"
	with open(args.output_file, 'w') as of:
		of.write(f"Dataset: {data_info}\n\n")
		of.write(f"CT x-y spacing counts: {ct_xy_spacing_counts}\n")
		of.write(f"CT z spacing counts: {ct_z_spacing_counts}\n")
		of.write(f"CT x-y size counts: {ct_xy_size_counts}\n")
		of.write(f"CT n-slices counts: {ct_n_slices_counts}\n\n")
		of.write(f"PET x-y spacing counts: {pet_xy_spacing_counts}\n")
		of.write(f"PET z spacing counts: {pet_z_spacing_counts}\n")
		of.write(f"PET x-y size counts: {pet_xy_size_counts}\n")
		of.write(f"PET n-slices counts: {pet_n_slices_counts}")


# ------------------------------------------------
if __name__ == '__main__':
	args = get_args()
	main(args)