# Data exploration notebooks, conversion scripts and utilities for PET-CT data

## Steps involved

### Preparing the data

1. Set a required physical volume size starting from the top (of the head) for the scans. Physical volume used is (450 x 450 x 300) mm3 in the (W,H,D) format. Generate bounding boxes of this size for each image and store them in a CSV file. Codename used for the outputs (and related items) of this step is "crFH" (crop with Full Head).

2. Resample to a specific resolution. The voxel spacing used is 1mm x 1mm x 3mm, i.e. inplane resolution of 1mm x 1mm and a slice thickness of 3mm. Codename used for the outputs (and related items) of this step is "rs113" (resampled to 1x1x3).



------------

## Libraries for IO and data processing

- For DICOM: [Pydicom](https://pydicom.github.io/)
- For multiple formats: [SimpleITK](https://simpleitk.org/) - Complete toolkit for N-dim scientific image-processing


## Software tools

- [3D Slicer](https://www.slicer.org/)


## Resources

- Doc: [DICOM format information for every image modality](https://dicom.innolitics.com/ciods/ct-image)
- Doc: [SimpleITK basic concepts](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html)
- Repo: [SimpleITK tutorial notebooks](https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks)

-------

- Article: [Medical Image Coordinates](https://theaisummer.com/medical-image-coordinates/)
- Article: [Medical Images In python (Computed Tomography)](https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography)
- Article: [A Comprehensive Guide To Visualizing and Analyzing DICOM Images in Python](https://medium.com/@hengloose/a-comprehensive-starter-guide-to-visualizing-and-analyzing-dicom-images-in-python-7a8430fcb7ed)
- Article: [Medical Image Analysis with Deep Learning , Part 4](https://www.kdnuggets.com/2017/07/medical-image-analysis-deep-learning-part-4.html)