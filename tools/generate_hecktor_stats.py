import os
import numpy as np


import SimpleITK as sitk


def main(hecktor_dir)

	train_data_dir = f"{hecktor_dir}/hecktor_train/hecktor_nii"
	patient_ids = sorted(os.listdir(train_data_dir))


	ct_xy_spacing_counts = {}
	ct_z_spacing_counts = {}
	ct_xy_size_counts = {}
	ct_n_slices_counts = {}

	pet_xy_spacing_counts = {}
	pet_z_spacing_counts = {}
	pet_xy_size_counts = {}
	pet_n_slices_counts = {}


	for p_id in tqdm(patient_ids):

	    # For CT --
	    ct_img_path = f"{train_data_dir}/{p_id}/{p_id}_ct.nii.gz"
	    ct_sitk = sitk.ReadImage(ct_img_path)

	    spacing = ct_sitk.GetSpacing() # Spacing: (W,H,D)
	    if round(spacing[0], 2) not in ct_xy_spacing_counts.keys():
	        ct_xy_spacing_counts[round(spacing[0], 2)] = 1
	    else:
	        ct_xy_spacing_counts[round(spacing[0], 2)] += 1

	    size = ct_sitk.GetSize() # Size = (W,H,D)
	    if size[0] not in ct_xy_size_counts.keys():
	        ct_xy_spacing_counts[size[0]] = 1
	    else:
	        ct_xy_spacing_counts[size[0]] += 1


	    # For PET --
	    pet_img_path = f"{train_data_dir}/{p_id}/{p_id}_pt.nii.gz"
	    pet_sitk = sitk.ReadImage(pet_img_path)

	    spacing = pet_sitk.GetSpacing() # Spacing: (W,H,D)
	    if round(spacing[0], 2) not in pet_xy_spacing_counts.keys():
	        pet_xy_spacing_counts[round(spacing[0], 2)] = 1
	    else:
	        pet_xy_spacing_counts[round(spacing[0], 2)] += 1

	    size = pet_sitk.GetSize() # Size = (W,H,D)
	    if size[0] not in pet_xy_size_counts.keys():
	        pet_xy_spacing_counts[size[0]] = 1
	    else:
	        pet_xy_spacing_counts[size[0]] += 1


	print("CT x-y spacing counts:", ct_xy_spacing_counts)
	print("CT z spacing counts:", ct_z_spacing_counts)
	print("CT x-y size counts:", ct_xy_size_counts)
	print("CT n-slices counts:", ct_n_slices_counts)

	print("PET x-y spacing counts:", pet_xy_spacing_counts)
	print("PET z spacing counts:", pet_z_spacing_counts)
	print("PET x-y size counts:", pet_xy_size_counts)
	print("PET n-slices counts:", pet_n_slices_counts)



	if __name__ == '__main__':
		hecktor_dir = "../data/HECKTOR"
		#stats_output_dir = "./outputs/"

		main()