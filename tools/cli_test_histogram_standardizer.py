import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk


DEFAULT_DATA_DIR = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
DEFAULT_PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
DEFAULT_OUTPUT_DIR = "../hecktor_meta"
DEFAULT_MODALITY = "PET"

# Histogram related constants
DEFAULT_QUANTILES_CUTOFF_SUV =  (0, 1)
DEFAULT_QUANTILES_CUTOFF_HU =  (0, 1)

CLIP_RANGE_SUV = [0,20]
CLIP_RANGE_HU = [-150, 150]


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        type=str,
                        default=DEFAULT_DATA_DIR,
                        help="Directory containing patient folders"
                        )
    parser.add_argument("--patient_id_file",
                        type=str,
                        default=DEFAULT_PATIENT_ID_FILE,
                        help="Patient ID file"
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default=DEFAULT_OUTPUT_DIR,
                        help="Output directory"
                        )
    parser.add_argument("--modality",
                        type=str,
                        required=True,
                        default=DEFAULT_MODALITY,
                        help="'PET' or 'CT'"
                        )
    args = parser.parse_args()
    return args


def main(args):

    data_dir = args.data_dir
    patient_id_file = args.patient_id_file
    modality = args.modality
    output_dir = args.output_dir

    with open(patient_id_file, 'r') as pf:
        patient_IDs = [p_id for p_id in pf.read().split("\n") if p_id != ""]

    # Landmarks
    landmarks_path = f"../hecktor_meta/hist_landmarks_{modality}.txt"
    landmarks = np.loadtxt(landmarks_path)

    print("Modality:", modality)
    for p_id in tqdm(patient_IDs):

        if modality == 'PET':
            image_sitk = sitk.ReadImage(f"{data_dir}/{p_id}_pt.nii.gz")
            clip_range = CLIP_RANGE_SUV
            quantiles_cutoff = DEFAULT_QUANTILES_CUTOFF_SUV
        elif modality == 'CT':
            image_sitk = sitk.ReadImage(f"{data_dir}/{p_id}_ct.nii.gz")
            clip_range = CLIP_RANGE_HU
            quantiles_cutoff = DEFAULT_QUANTILES_CUTOFF_HU

        image_np = sitk.GetArrayFromImage(image_sitk)

        # Clip the image first to keep intensities in the required range
        image_np = np.clip(image_np, clip_range[0], clip_range[1])

        # Percentiles of the image
        percentiles_cutoff = 100 * np.array(quantiles_cutoff)
        quartiles = np.arange(25, 100, 25).tolist()
        deciles = np.arange(10, 100, 10).tolist()
        all_percentiles = list(percentiles_cutoff) + quartiles + deciles
        percentiles = sorted(set(all_percentiles))
        percentiles = np.array(percentiles)
        if modality == 'CT':  image_np = np.clip(image_np, -1000, image_np.max())
        percentile_values = np.percentile(image_np, percentiles)

        plt.plot(percentile_values, landmarks)

    #for pc in percentile_values:
    #   plt.plot([pc, pc], [landmarks.min(), landmarks.max()], color='black', alpha=0.5, linestyle='dashed')
    plt.xlabel("Image scale")
    plt.ylabel("Standard scale")
    plt.title(f"Histogram transform - {modality}")
    plt.savefig(f"{output_dir}/{modality}_histogram_transform.png")
    plt.show()



if __name__ == '__main__':
    args = get_args()
    main(args)