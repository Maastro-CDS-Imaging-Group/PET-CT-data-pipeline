"""
# TODO
	x Add multichannel representation
	- Cross validation support
"""

import sys, random

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torchio
import SimpleITK as sitk

sys.path.append("../")
from data_utils.conversion import *
import data_utils.transforms as transforms


# Constants
AUG_PROBABILITY = 0.5


class HECKTORPETCTDataset(torch.utils.data.Dataset):

	def __init__(self, data_dir, patient_id_filepath, mode='train', preprocessor=None, input_representation='separate volumes', augment_data=False):

		self.data_dir = data_dir
		with open(patient_id_filepath, 'r') as pf:
			self.patient_ids = [p_id for p_id in pf.read().split('\n') if p_id != '']

		# CHGJ(55), CHMR(18) and CHUS(72) for training. CHUM(56) for validation
		self.mode = mode
		if self.mode == 'train':
			self.patient_ids = [p_id for p_id in self.patient_ids if 'CHUM' not in p_id]
		elif self.mode == 'validation':
			self.patient_ids = [p_id for p_id in self.patient_ids if 'CHUM' in p_id]

		self.input_representation = input_representation

		self.spacing_dict = {'xy spacing': 1.0, 'slice thickness': 3.0}
		self.preprocessor = preprocessor
		if self.preprocessor is None:
			raise Exception("Specify the preprocessor")
		preprocessor.set_spacing(self.spacing_dict)

		# Augmentation config
		self.augment_data = augment_data
		self.affine_matrix = np.array(      # Affine matrix representing the (1,1,3) spacing as scaling
		                              [
		                               [1,0,0,0],
                                       [0,1,0,0],
                  	                   [0,0,3,0],
                                       [0,0,0,1]
                                      ]
                                     )
		self.torchio_oneof_transform = None
		self.PET_stretch_transform = None
		if self.augment_data:
			self.torchio_oneof_transform, self.PET_stretch_transform = transforms.build_transforms()


	def __len__(self):
		return len(self.patient_ids)


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

		# Smooth PET and CT
		PET_np = self.preprocessor.smoothing_filter(PET_np, modality='PET')
		CT_np = self.preprocessor.smoothing_filter(CT_np, modality='CT')

		# Standardize the intensity scale
		PET_np = self.preprocessor.standardize_intensity(PET_np, modality='PET')
		CT_np = self.preprocessor.standardize_intensity(CT_np, modality='CT')

		# Data augmentation
		if self.augment_data:
			if random.random() < AUG_PROBABILITY:
				PET_np, CT_np, GTV_labelmap_np = self.apply_transform(PET_np, CT_np, GTV_labelmap_np)

		# Rescale intensities to [0,1] range
		# PET_np = self.preprocessor.rescale_to_unit_range(PET_np)
		# CT_np = self.preprocessor.rescale_to_unit_range(CT_np)

		# Construct the sample dict -- Convert to tensor and change dim ordering to (D,H,W)
		if self.input_representation == 'separate volumes':
			# Provide PET and CT as 2 separate input tensors, each of shape (1,D,H,W). GTV mask will be of shape (D,H,W)
			sample_dict = {'PET': np2tensor(PET_np).permute(2,1,0).unsqueeze(dim=0),
	                       'CT': np2tensor(CT_np).permute(2,1,0).unsqueeze(dim=0),
	                       'GTV labelmap': np2tensor(GTV_labelmap_np).permute(2,1,0)
			              }

		elif self.input_representation == 'multichannel volume':
			# Pack PET and CT into a single input tensor of shape (2,D,H,W). GTV mask will be of shape (D,H,W)
			PET_tnsr = np2tensor(PET_np).permute(2,1,0)
			CT_tnsr = np2tensor(CT_np).permute(2,1,0)
			sample_dict = {'PET-CT': torch.stack([PET_tnsr, CT_tnsr], dim=0),
	                       'GTV labelmap': torch.from_numpy(GTV_labelmap_np).permute(2,1,0)
						  }

		return sample_dict


	def apply_transform(self, PET_np, CT_np, GTV_labelmap_np):
		r = random.random()
		if  r < 0.75:
			# Apply one of the 3 TorchIO spatial transforms
			subject = self._create_torchio_subject(PET_np, CT_np, GTV_labelmap_np)
			subject = self.torchio_oneof_transform(subject)
			PET_np = subject['PET'].numpy().squeeze()
			CT_np = subject['CT'].numpy().squeeze()
			GTV_labelmap_np = subject['GTV labelmap'].numpy().squeeze()
		else:
			# PET intensity stretching
			PET_np = self.PET_stretch_transform(PET_np)
		return PET_np, CT_np, GTV_labelmap_np

	def _create_torchio_subject(self, PET_np, CT_np, GTV_labelmap_np):
		PET_tio = torchio.Image(tensor=np2tensor(PET_np).unsqueeze(dim=0), type=torchio.INTENSITY, affine=self.affine_matrix)
		CT_tio = torchio.Image(tensor=np2tensor(CT_np).unsqueeze(dim=0), type=torchio.INTENSITY, affine=self.affine_matrix)
		GTV_labelmap_tio = torchio.Image(tensor=np2tensor(GTV_labelmap_np).unsqueeze(dim=0), type=torchio.LABEL, affine=self.affine_matrix)
		subject_dict = {'PET': PET_tio, 'CT': CT_tio, 'GTV labelmap': GTV_labelmap_tio}
		subject = torchio.Subject(subject_dict)
		return subject



if __name__ == '__main__':
	data_dir = "/home/zk315372/Chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
	patient_id_filepath = "../hecktor_meta/patient_IDs_train.txt"

	dataset = HECKTORPETCTDataset(data_dir,
		                          patient_id_filepath,
		                          mode='train',
		                          input_representation='separate volumes',
		                          augment_data=False)

	from data_utils.preprocessing import Preprocessor
	preprocessor = Preprocessor()
	dataset.set_preprocessor(preprocessor)

	sample = dataset[0]