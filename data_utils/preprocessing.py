from scipy.ndimage import gaussian_filter


class Preprocessor():

	def __init__(self, spacing_dict, PET_smooth_sigma_mm=2.0, CT_smooth_sigma_mm=1.0):
		self.spacing_dict = spacing_dict
		self.PET_smooth_sigma_mm = PET_smooth_sigma_mm
		self.CT_smooth_sigma_mm = CT_smooth_sigma_mm


	def smoothing_filter(self, image_np, modality):
		if modality == 'PET': smoothing_sigma_mm = PET_smooth_sigma_mm
		elif modality == 'CT': smoothing_sigma_mm = CT_smooth_sigma_mm

		sigma = (self.smoothing_sigma_mm / self.spacing_dict['xy spacing'],
		         self.smoothing_sigma_mm / self.spacing_dict['xy spacing'],
		         self.smoothing_sigma_mm / self.spacing_dict['slice thickness'])

		image_np = gaussian_filter(image_np, sigma=sigma)
		return image_np



# Smoothing filter


# Intensity normalization / scaling