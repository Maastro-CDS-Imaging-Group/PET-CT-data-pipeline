"""
Script to convert the original Head-Neck-PET-CT dataset into a more useable version.
File types to be converted: 
    1. CT DICOM series to NRRD
    2. PET DICOM series to NRRD
    3. RTSTRUCT DICOM file for CT to NRRD with matching volumes
    4. RTSTRUCT DICOM file for PET to NRRD with matching volumes
"""

import os

import numpy as np

import pydicom
import SimpleITK as sitk

DATA_DIR = "../Data/Head-Neck-PET-CT/"
