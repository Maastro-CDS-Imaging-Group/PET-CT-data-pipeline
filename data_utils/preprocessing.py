from scipy.ndimage import gaussian_filter


class Preprocessor():

	def __init__(self, spacing_dict={'xy spacing':1, 'slice thickness':3},
		         smooth_sigma_mm={'PET': 2.0, 'CT': 1.0},
		         standardization_method='clipping',
		         clipping_range={'PET': [0,20], 'CT': [-150,150]},
		         histogram_landmarks_path={'PET': None, 'CT': None}
		         ):

		# Smoothing params
		self.spacing_dict = spacing_dict
		self.smooth_sigma_mm = smooth_sigma_mm

		# Standardization params
		self.standardization_method = standardization_method
		self.clipping_range = clipping_range
		self.histogram_landmarks_path = histogram_landmarks_path


	def smoothing_filter(self, image_np, modality):
		sigma_mm = self.smooth_sigma_mm[modality]
		sigma = (sigma_mm / self.spacing_dict['xy spacing'],
		         sigma_mm / self.spacing_dict['xy spacing'],
		         sigma_mm / self.spacing_dict['slice thickness'])
		image_np = gaussian_filter(image_np, sigma=sigma)
		return image_np


	def standardize_intensity(self, image_np, modality):
		if self.standardization_method == 'clipping':
			image_np = self._clip(image_np, modality)
		elif self.standardization_method == 'histogram':
			image_np = self._histogram_transform(image_np, modality)
		return image_np

	def _clip(self, image_np, modality):
		clipping_range = self.clipping_range[modality]
		image_np[image_np < clipping_range[0]] = clipping_range[0]
		image_np[image_np > clipping_range[1]] = clipping_range[1]
		return image_np

	def _histogram_transform(self, image_np, modality):
		# TODO
		pass


	def scale_to_unit_range(self, image_np):
		image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
		return image_np