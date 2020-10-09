import argparse
import numpy as np
from torchio.transforms import HistogramStandardization


DEFAULT_DATA_DIR = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
DEFAULT_PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
DEFAULT_OUTPUT_DIR = "../hecktor_meta"
DEFAULT_MODALITY = "PET"


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
                        rewuired=True,
                        default=DEFAULT_MODALITY,
                        help="'PET' or 'CT'"
                        )
    args = parser.parse_args()
    return args


def main(args):

    with open(args.patient_id_file, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split("\n") if p_id != ""]

    if args.modality == 'PET':
        # PET
        patient_PET_paths = [f"{args.data_dir}/{p_id}_pt.nii.gz" for p_id in patient_ids]
        PET_landmarks = HistogramStandardization.train(patient_PET_paths, cutoff=(0.02, 0.98))
        np.save(f"{args.output_dir}/hist_landmarks_PET.npy", PET_landmarks)

    elif args.modality == 'CT':
        # CT
        patient_CT_paths = [f"{args.data_dir}/{p_id}_ct.nii.gz" for p_id in patient_ids]
        CT_landmarks = HistogramStandardization.train(patient_CT_paths, cutoff=(0.02, 0.98))
        np.save(f"{args.output_dir}/hist_landmarks_CT.npy", CT_landmarks)


if __name__ == '__main__':
    args = get_args()
    main(args)
