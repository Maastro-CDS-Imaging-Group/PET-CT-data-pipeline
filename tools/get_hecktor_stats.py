"""
Obtain counts for voxel spacing and array sizes for all *original* CT and PET images form HECKTOR dataset.
This information is mainly to be used for getting an overview of the original data and for finding the best voxel spacing for resampling. 
"""

import os
import numpy as np
from tqdm import tqdm
import json

import matplotlib.pyplot as plt

import SimpleITK as sitk


def main(hecktor_dir, dataset='train', output_dir=None):

	if dataset == 'train':
		data_dir = f"{hecktor_dir}/hecktor_train/hecktor_nii"
	elif dataset == 'test':
		data_dir = f"{hecktor_dir}/hecktor_test/hecktor_nii_test"
	
	patient_ids = sorted(os.listdir(data_dir))
	print("Patients found:", len(patient_ids))

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
		ct_img_path = f"{data_dir}/{p_id}/{p_id}_ct.nii.gz"
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
		pet_img_path = f"{data_dir}/{p_id}/{p_id}_pt.nii.gz"
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
	output_stats_filepath = f"{output_dir}/hecktor_{dataset}_stats.txt"
	with open(output_stats_filepath, 'w') as of:
		of.write(f"Dataset: {dataset}\n\n")
		of.write(f"CT x-y spacing counts: {ct_xy_spacing_counts}\n")
		of.write(f"CT z spacing counts: {ct_z_spacing_counts}\n")
		of.write(f"CT x-y size counts: {ct_xy_size_counts}\n")
		of.write(f"CT n-slices counts: {ct_n_slices_counts}\n\n")
		of.write(f"PET x-y spacing counts: {pet_xy_spacing_counts}\n")
		of.write(f"PET z spacing counts: {pet_z_spacing_counts}\n")
		of.write(f"PET x-y size counts: {pet_xy_size_counts}\n")
		of.write(f"PET n-slices counts: {pet_n_slices_counts}")



if __name__ == '__main__':

	hecktor_dir = "/home/zk315372/Chinmay/Datasets/HECKTOR"

	# For train data
	main(hecktor_dir, dataset='train', output_dir="./outputs")

	# For test data
	main(hecktor_dir, dataset='test', output_dir="./outputs")
	