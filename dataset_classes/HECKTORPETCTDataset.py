"""
# TODO Add data augmentation
"""

import sys, random

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import SimpleITK as sitk

from data_utils.conversion import *


class HECKTORPETCTDataset(torch.utils.data.Dataset):

	def __init__(self, data_dir, patient_id_filepath, mode='train', input_representation='separate volumes', data_augment=False):

		self.data_dir = data_dir
		with open(patient_id_filepath, 'r') as pf:
			self.patient_ids = [p_id in pf.read().split('\n') if p_id != '']

		self.mode = mode
		self.input_representation = input_representation

		self.xy_spacing = 1.0
		self.slice_thickness = 3.0

		self.preprocessor = None
		self.data_augment = data_augment


	# Getters
	def get_spacing():
		spacing_dict = {'xy spacing': self.xy_spacing, 'slice thickness': self.slice_thickness}
		return spacing_dict


	# Setters
	def set_preprocessor(self, preprocessor):
		self.preprocessor = preprocessor


	def __getitem__(self, idx):
		p_id = self.patient_ids[idx]

		# Read data files into sitk images -- (W,H,D) format
		PET_sitk = sitk.ReadImage(f"{self.data_dir}/{p_id}_pt.nii.gz")
		CT_sitk = sitk.ReadImage(f"{self.data_dir}/{p_id}_ct.nii.gz")
		GTV_labelmap_sitk = sitk.ReadImage(f"{self.data_dir}/{p_id}_ct_gtvt.nii.gz")

		# Convert to ndarrays -- Keep the (W,H,D) dim ordering
		PET_np = sitk2np(PET_sitk, keep_whd_ordering=True)
		CT_np = sitk2np(CT_sitk, keep_whd_ordering=True)
		GTV_labelmap_np = sitk2np(GTV_labelmap_sitk, keep_whd_ordering=True)

		# Check memory usage
		print(sys.getsizeof(PET_sitk)) / 1024
		print(sys.getsizeof(PET_np)) / 1024
		print(PET_np.dtype)

		# Smooth PET and CT
		PET_np = self.preprocessor.smoothing_filter(PET_np, modality='PET')
		CT_np = self.preprocessor.smoothing_filter(CT_np, modality='CT')

		# Data augmentation
		if self.data_augment:
			if random.random() > 0.5:
				PET_np, CT_np, GTV_labelmap_np = augment_transform(PET_np, CT_np, GTV_labelmap_np)



	def augment_transform(PET_np, CT_np, GTV_labelmap_np):
		r = random.randint(0,5)

		# Random rotation

		# Crop and resize

		# Elastic distortion

		# PET intensity stretching
