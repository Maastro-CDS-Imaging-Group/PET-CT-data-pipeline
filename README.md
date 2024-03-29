# Notebooks, scripts and utilities for PET-CT data processing

For segmentation of Head-and-Neck gross tumor volume using deep learning-based PET-CT fusion models.


------------

## Steps involved
### Preparing the data

1. Set a required physical volume size starting from the top (of the head) for the scans. Physical volume used is (450 x 450 x 300) mm3 in the (W,H,D) format. This size is chosen so that majority of the Head-and-Neck region is included. The generated bounding boxes of this size for each image are written in a CSV file. 
	```
	$ python cli_generate_bboxes.py  --source_dir ...   
	                                 --bbox_filepath ...  
	                                 --output_phy_size 450 450 300
	``` 

2. Crop the images according to their bounding box's physical coordinates and resample to the specified spacing/resolution. The voxel spacing used is 1mm x 1mm x 3mm, i.e. inplane resolution of 1mm x 1mm and a slice thickness of 3mm. A cropped+resampled version of the dataset is created and written on the disk. 
	```
	$ python cli_hktr_crop_and_resample.py  --source_dir ...  
	                                        --target_dir ...  
	                                        --bbox_filepath ...  
	                                        --new_spacing 1 1 3  
	                                        --cores 24  
	                                        --order 3
	```

Codename used for the outputs (and related items) of this step is "crFH_rs113" (cropped keeping Full Head, resampled to 1x1x3). The images thus obtained have a physical volume of (450 x 450 x 300) mm3 and an array size of (450 x 450 x 100) voxels.

### Patient Dataset
Defines a dataset which fetches and returns paired input-ouput full volumes of a given patient (index). Performs modality-specific preprocessing and transforms.
Subclass of torch.utils.data.Dataset.

Processing involved:

1. Smoothing 
	- Mainly for PET, to remove/reduce PSF reconstruction induced overshoot artifacts.

2. Intensity standardization  
	- 2 options: Intensity clipping or histogram standardization
	- Clipping for CT is set to [-150,150] HU and for PET [0,20] SUV by default.
	- Clipping + Histogram standardization [[Histogram paper](https://ieeexplore.ieee.org/document/836373) | [TorchIO Histogram standardization](https://torchio.readthedocs.io/transforms/preprocessing.html#histogramstandardization)]: Computing a mean histogram using the training samples, and distorting the histograms of the images to match this mean histogram using piece-wise linear contrast adjustment. To be done separately for PET and CT, obviously. 

3. Augmentation transform 
	- Spatial: Random rotation(+-10 degrees), scaling(+-15%), elastic distortion.
	- Intensity: Contrast stretching in PET between 30 and 95 percentile SUV range. 
	- One of these 4 applied with equal probability.

4. Rescaling intensities to [0,1] range
	Min-max normalization, where min and max values are obtained from the volume.

### Patch Queue
Combined with a patch sampler, the patch queue creates, stores and returns randomly sampled paired input-output patches of given size. Code adapted from [TorchIO Queue](https://torchio.readthedocs.io/data/patch_training.html#id1) source code. The PatchQueue class is derived from torch.data.utils.Dataset. See the GIF on the linked page for working mechanism.

### Patch loader
Used with the patch queue to create batches of patches for training. Regular torch dataloader instance.


------------

## Libraries for IO and data processing
- For DICOM: [Pydicom](https://pydicom.github.io/)
- For multiple formats: [SimpleITK](https://simpleitk.org/) - Complete toolkit for N-dim scientific image-processing


------------

## Software tools
- [3D Slicer](https://www.slicer.org/)


------------

## Resources
### Documentation
- [DICOM format information for every image modality](https://dicom.innolitics.com/ciods/ct-image)
- [SimpleITK basic concepts](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html)

### Code repositories
- [SimpleITK tutorial notebooks](https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks)

### Online articles
- [Medical Image Coordinates](https://theaisummer.com/medical-image-coordinates/)
- [Medical Images In python (Computed Tomography)](https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography)
- [A Comprehensive Guide To Visualizing and Analyzing DICOM Images in Python](https://medium.com/@hengloose/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed)
- [Medical Image Analysis with Deep Learning , Part 4](https://www.kdnuggets.com/2017/07/medical-image-analysis-deep-learning-part-4.html)
