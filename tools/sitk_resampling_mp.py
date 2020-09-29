import os
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

import multiprocessing as mp

import SimpleITK as sitk



SITK_INTERPOLATOR_DICT = {'nearest': sitk.sitkNearestNeighbor,
                          'linear': sitk.sitkLinear,
}

DEFAULT_VOXEL_VALUE = -1000


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", 
                        type=str, 
                        default="../../../Datasets/HECKTOR/hecktor_train/hecktor_nii",
                        help="Directory containing patient folders"
                        )

    parser.add_argument("--target_dir", 
                        type=str, 
                        default="../../../Datasets/HECKTOR/hecktor_train/resampled_hecktor_nii",
                        help="Directory that will contain patient folders"
                        )

    parser.add_argument("--new_spacing", 
                        type=int,
                        nargs=3,
                        default=[1,1,3],
                        help="New voxel spacing format - (W,H,D)"
                        )

    args = parser.parse_args()
    return args


def resample_sitk_image(sitk_image, new_spacing, sitk_interpolator):
    # Get original image's info
    orig_image_info = {'size': sitk_image.GetSize(), 
                       'spacing': sitk_image.GetSpacing(), 
                       'origin': sitk_image.GetOrigin(), 
                       'direction': sitk_image.GetDirection(), 
                       'pixel id value': sitk_image.GetPixelIDValue()
    }

    # Calculate new size
    orig_size = np.array(orig_image_info['size'])
    orig_spacing = np.array(orig_image_info['spacing'])

    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.ceil(new_size).astype(np.int)
    new_size = [int(s) for s in new_size]

    # Apply resampling
    sitk_image_resampled = sitk.Resample(sitk_image, 
                                         new_size,
                                         sitk.Transform(),
                                         sitk_interpolator,
                                         orig_image_info['origin'],
                                         new_spacing,
                                         orig_image_info['direction'],
                                         DEFAULT_VOXEL_VALUE,
                                         orig_image_info['pixel id value'])

    return sitk_image_resampled


def process_patient(p_id):
    print("Processing patient:", p_id)
    source_patient_dir = source_dir / Path(p_id)

    # CT
    ct_image_path = source_patient_dir / Path(f"{p_id}_ct.nii.gz")
    ct_sitk = sitk.ReadImage(str(ct_image_path))
    sitk_interpolator = SITK_INTERPOLATOR_DICT['linear']
    ct_resampled_sitk = resample_sitk_image(ct_sitk, new_spacing, sitk_interpolator)
    
    # PET
    pet_sitk = sitk.ReadImage(f"{source_patient_dir}/{p_id}_pt.nii.gz")
    sitk_interpolator = SITK_INTERPOLATOR_DICT['linear']
    pet_resampled_sitk = resample_sitk_image(pet_sitk, new_spacing, sitk_interpolator)

    # GTV mask
    gtv_sitk = sitk.ReadImage(f"{source_patient_dir}/{p_id}_ct_gtvt.nii.gz")
    sitk_interpolator = SITK_INTERPOLATOR_DICT['nearest']
    gtv_resampled_sitk = resample_sitk_image(gtv_sitk, new_spacing, sitk_interpolator)

    # Write into target directory
    target_patient_dir = target_dir / Path(p_id)
    target_patient_dir.mkdir(parents=True, exist_ok=True)

    ct_resampled_path = target_patient_dir / Path(f"{p_id}_ct.nrrd")
    sitk.WriteImage(ct_resampled_sitk, str(ct_resampled_path), useCompression=True)

    pet_resampled_path = target_patient_dir / Path(f"{p_id}_pt.nrrd")
    sitk.WriteImage(pet_resampled_sitk, str(pet_resampled_path), useCompression=True)
    
    gtv_resampled_path = target_patient_dir / Path(f"{p_id}_ct_gtvt.nrrd")
    sitk.WriteImage(gtv_resampled_sitk, str(gtv_resampled_path), useCompression=True)



if __name__ == '__main__':

    args = get_args()

    new_spacing = [float(s) for s in args.new_spacing]
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)

    patient_ids = [str(dir_path).split('/')[-1] for dir_path in list(source_dir.glob("*/"))]
    print("Total patients found:", len(patient_ids))

    processes = []
    for p_id in patient_ids:
        p = mp.Process(target=process_patient, args=(p_id,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

        



