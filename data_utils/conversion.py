import numpy as np
import torch
import SimpleITK as sitk



# Between SimpleITK Image and Numpy ndarray
def sitk2np(image_sitk, keep_whd_ordering=True):
	image_np = sitk.GetArrayFromImage(image_sitk)
	if keep_whd_ordering:
		image_np = image_np.transpose((2,1,0))
	return image_np


def np2sitk(image_np, np_in_whd_order=True):
	if np_in_whd_order:
		image_np = image_np.transpose((2,1,0)) # Convert to (D,H,W)
	image_sitk = sitk.GetImageFromArray(image_np)
	return image_sitk



# Between Torch tensor and Numpy ndarray
def tensor2np(image_tensor):
	image_np = image_tensor.cpu().numpy()
	return image_np


def np2tensor(image_np):
	image_tensor = torch.from_numpy(image_np)
	return image_tensor



# Between SimpleITK Image and Torch tensor
def sitk2tensor(image_sitk):
	image_np = sitk.GetArrayFromImage(image_sitk)
	image_tensor = torch.from_numpy(image_np)
	return image_tensor


def tensor2sitk(image_tensor):
	image_np = image_tensor.cpu().numpy()
	image_sitk = sitk.GetImageFromArray(image_np)
	return image_sitk