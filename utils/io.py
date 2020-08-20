import os

import numpy as np

import pydicom
import SimpleITK as sitk

def read_dcm_series(series_dir):
    """
    Read a DICOM series into a sitk image.
    Parameters
        series_dir: Directory containing a DICOM series
    Returns
        sitk_image: SimpleITK Image object
        meta: Python dictionary containing selected meta-data
    """
    if len(os.listdir(series_dir)) > 1:
        reader = sitk.ImageSeriesReader()
        dcm_file_paths = reader.GetGDCMSeriesFileNames(series_dir)
        sitk_image = sitk.ReadImage(dcm_file_paths)
        dicom_data = pydicom.read_file(dcm_file_paths[0], stop_before_pixels=True)
    else:
        file_name = os.listdir(series_dir)[0]
        sitk_image = sitk.ReadImage(series_dir + file_name)
        dicom_data = pydicom.read_file(series_dir + file_name, stop_before_pixels=True)

    meta = {}

    meta['PatientID'] = dicom_data.PatientID
    meta['StudyDescription'] = dicom_data.StudyDescription
    meta['StudyInstanceUID'] = dicom_data.StudyInstanceUID
    meta['SeriesDescription'] = dicom_data.SeriesDescription
    meta['SeriesInstanceUID'] = dicom_data.SeriesInstanceUID
    meta['Modality'] = dicom_data.Modality

    meta['Pixel spacing'] = sitk_image.GetSpacing()
    meta['Width'] = sitk_image.GetWidth()
    meta['Height'] = sitk_image.GetHeight()
    meta['Depth'] = sitk_image.GetDepth()
    meta['Direction'] = sitk_image.GetDirection()

    return sitk_image, meta



if __name__ == '__main__':
        
    study_dir_1 = "./Data/Head-Neck-PET-CT/HN-CHUS-009/08-27-1885-65886/"    
    series_dir_1 = study_dir_1 + "517120.000000-LOR-RAMLA-54001/"
    
    study_dir_2 = "./Data/Head-Neck-PET-CT/HN-CHUS-009/08-27-1885-TEP cancerologique-77284/"
    series_dir_2 = study_dir_2 + "3.000000-Merged-51326/"
    
    print("Series 1")
    sitk_image_1, meta_1 = read_dcm_series(series_dir_1)
    for k,v in meta_1.items():
        print(k, ":", v)

    print("Series 2")
    sitk_image_2, meta_2 = read_dcm_series(series_dir_2)
    for k,v in meta_2.items():
        print(k, ":", v)