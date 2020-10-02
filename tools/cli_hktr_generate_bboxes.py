"""

Script to automatically generate bounding boxes of given *physical* size by performing 
brain segmentation in PET using SUV thresholding.

Modified to contain full head (FH).

Adapted from github.com/voreille/hecktor

"""

import os, argparse

from tqdm import tqdm

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import label


# Constants
DEFAULT_SOURCE_DIR = "../../../Datasets/HECKTOR/hecktor_train/hecktor_nii"
DEFAULT_BB_FILEPATH = "../heck_meta/bboxes_train_FH.csv"
DEFAULT_PHY_SIZE = [450.0, 450.0, 300.0] 


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", 
                        type=str, 
                        default=DEFAULT_SOURCE_DIR,
                        help="Directory containing patient folders"
                        )

    parser.add_argument("--bbox_filepath", 
                        type=str, 
                        default=DEFAULT_BB_FILEPATH,
                        help="Output CSV file that would contain bbox coordinates"
                        )

    parser.add_argument("--output_phy_size", 
                        type=float, 
                        nargs=3,
                        default=DEFAULT_PHY_SIZE,
                        help="Required physical size for the output images in mm -- (W,H,D) format"
                        )

    args = parser.parse_args()
    return args



def bbox_auto(np_pt,
              px_spacing_pt,
              px_origin_pt,
              output_shape=(144, 144, 144),
              th=3):
    """Find a bounding box automatically based on the SUV
    Arguments:
        vol_pt {numpy array} -- The PET volume on which to compute the bounding box
        px_spacing_pt {tuple of float} -- The spatial resolution of the PET volume 
        px_origin_pt {tuple of float} -- The spatial position of the first voxel in the PET volume 
    Keyword Arguments:
        shape {tuple} -- The ouput size of the bounding box in millimeters
        th {float} -- [description] (default: {3})
    Returns:
        [type] -- [description]
    """

    np_pt = np_pt.transpose((2,1,0)) # Custom hack

    output_shape_pt = tuple(e1 / e2 for e1, e2 in zip(output_shape, px_spacing_pt))
    # Gaussian smooth
    np_pt_gauss = gaussian_filter(np_pt, sigma=3)
    # auto_th: based on max SUV value in the top of the PET scan
    #auto_th = np.max(np_pt[:, :, np.int(np_pt.shape[2] * 2 // 3):]) / 4
    #print('auto_th = ', auto_th)
    # OR fixed threshold
    np_pt_thgauss = np.where(np_pt_gauss > th, 1, 0)
    # Find brain as biggest blob AND not in lowest third of the scan
    labeled_array, _ = label(np_pt_thgauss)
    try:
        np_pt_brain = labeled_array == np.argmax(
            np.bincount(labeled_array[:,:,np_pt.shape[2] * 2 // 3:].flat)[1:]) + 1
    except:
        tqdm.write('th too high?')
        # Quick fix just to pass for all cases
        th = 0.1
        np_pt_thgauss = np.where(np_pt_gauss > th, 1, 0)
        labeled_array, _ = label(np_pt_thgauss)
        np_pt_brain = labeled_array == np.argmax(
            np.bincount(labeled_array[:,:,np_pt.shape[2] * 2 // 3:].flat)[1:]) + 1
    # Find lowest voxel of the brain and box containing the brain
    # z = np.min(np.argwhere(np.sum(np_pt_brain, axis=(0, 1))))
    z = np.max(np.argwhere(np.sum(np_pt_brain, axis=(0, 1))))  # Altered to find the *highest* voxel of the brain instead
    y1 = np.min(np.argwhere(np.sum(np_pt_brain, axis=(0, 2))))
    y2 = np.max(np.argwhere(np.sum(np_pt_brain, axis=(0, 2))))
    x1 = np.min(np.argwhere(np.sum(np_pt_brain, axis=(1, 2))))
    x2 = np.max(np.argwhere(np.sum(np_pt_brain, axis=(1, 2))))

    # Center bb based on this brain segmentation
    zshift = 30 // px_spacing_pt[2]
    if z - (output_shape_pt[2] - zshift) < 0:
        zbb = (0, output_shape_pt[2])
    elif z + zshift > np_pt.shape[2]:
        zbb = (np_pt.shape[2] - output_shape_pt[2], np_pt.shape[2])
    else:
        zbb = (z - (output_shape_pt[2] - zshift), z + zshift)

    yshift = 30 // px_spacing_pt[1]
    if np.int((y2 + y1) / 2 - yshift - np.int(output_shape_pt[1] / 2)) < 0:
        ybb = (0, output_shape_pt[1])
    elif np.int((y2 + y1) / 2 - yshift -
                np.int(output_shape_pt[1] / 2)) > np_pt.shape[1]:
        ybb = np_pt.shape[1] - output_shape_pt[1], np_pt.shape[1]
    else:
        ybb = ((y2 + y1) / 2 - yshift - output_shape_pt[1] / 2,
               (y2 + y1) / 2 - yshift + output_shape_pt[1] / 2)

    if np.int((x2 + x1) / 2 - np.int(output_shape_pt[0] / 2)) < 0:
        xbb = (0, output_shape_pt[0])
    elif np.int((x2 + x1) / 2 - np.int(output_shape_pt[0] / 2)) > np_pt.shape[0]:
        xbb = np_pt.shape[0] - output_shape_pt[0], np_pt.shape[0]
    else:
        xbb = ((x2 + x1) / 2 - output_shape_pt[0] / 2,
               (x2 + x1) / 2 + output_shape_pt[0] / 2)

    z_pt = np.asarray(zbb)
    y_pt = np.asarray(ybb)
    x_pt = np.asarray(xbb)

    # In the physical dimensions
    z_abs = z_pt * px_spacing_pt[2] + px_origin_pt[2]
    y_abs = y_pt * px_spacing_pt[1] + px_origin_pt[1]
    x_abs = x_pt * px_spacing_pt[0] + px_origin_pt[0]
    
    bb = np.asarray((x_abs, y_abs, z_abs)).flatten()
    return bb



def main(args):
    source_dir = args.source_dir
    patient_ids = sorted(os.listdir(source_dir))

    bb_df = pd.DataFrame(
        columns=['PatientID', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2'])

    for p_id in tqdm(patient_ids):
        pet_sitk = sitk.ReadImage(f"{source_dir}/{p_id}/{p_id}_pt.nii.gz")
        pet_np = sitk.GetArrayFromImage(pet_sitk)
        spacing = pet_sitk.GetSpacing()
        origin = pet_sitk.GetOrigin()
        bb = bbox_auto(pet_np, spacing, origin, args.output_phy_size)
        bb_df = bb_df.append(
            {
                'PatientID': p_id,
                'x1': bb[0],
                'x2': bb[1],
                'y1': bb[2],
                'y2': bb[3],
                'z1': bb[4],
                'z2': bb[5],
            },
            ignore_index=True)
    bb_df.to_csv(args.bbox_filepath)


if __name__ == '__main__':
    args = get_args()
    main(args)