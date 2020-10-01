"""
Plot HU and SUV distributions for CT and PET respectively. 
Note: Distribution is only *after* resampling all images to common voxel spacing
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import SimpleITK as sitk


def main(hecktor_dir, dataset='train_full_res', output_dir=None):
    
    """
    dataset options: train_full_res, test_full_res, 
                     train_crop_res, test_crop_res
    """

	# if dataset == 'train':
	# 	data_dir = f"{hecktor_dir}/hecktor_train/hecktor_nii"
	# elif dataset == 'test':
	# 	data_dir = f"{hecktor_dir}/hecktor_test/hecktor_nii_test"
	
	patient_ids = sorted(os.listdir(data_dir))
	print("Patients found:", len(patient_ids))

	fig_ct, ax_ct = plt.subplots(figsize=(20,10))
	fig_pet, ax_pet = plt.subplots(figsize=(20,10))
	
	for p_id in tqdm(patient_ids):

		# For CT -------------------------------
		ct_img_path = f"{data_dir}/{p_id}/{p_id}_ct.nii.gz"
		ct_sitk = sitk.ReadImage(ct_img_path)
		
		# HU histogram
		ct_np = sitk.GetArrayFromImage(ct_sitk)
		ax_ct.hist(ct_np.flatten(), bins=np.linspace(-4000, 2000, 256), histtype='step')


		# For PET -------------------------------
		pet_img_path = f"{data_dir}/{p_id}/{p_id}_pt.nii.gz"
		pet_sitk = sitk.ReadImage(pet_img_path)

		# SUV histogram
		pet_np = sitk.GetArrayFromImage(pet_sitk)
		ax_pet.hist(pet_np.flatten(), bins=np.linspace(-5, 30, 256), histtype='step')


	# Save histogram plots
	ax_ct.set_xlabel("HU")
	ax_ct.set_ylabel("Number of voxels")
	output_hist_filepath_ct = f"{output_dir}/hecktor_{dataset}_ct_hist.png"
	fig_ct.savefig(output_hist_filepath_ct)

	ax_pet.set_xlabel("SUV")
	ax_pet.set_ylabel("Number of voxels")
	output_hist_filepath_pet = f"{output_dir}/hecktor_{dataset}_pet_hist.png"
	fig_pet.savefig(output_hist_filepath_pet)



if __name__ == '__main__':

	hecktor_dir = "/home/zk315372/Chinmay/Datasets/HECKTOR"

	# For train data
	main(hecktor_dir, dataset='train_full_res', output_dir="./outputs")

	# For test data
	main(hecktor_dir, dataset='test_full_res', output_dir="./outputs")
	