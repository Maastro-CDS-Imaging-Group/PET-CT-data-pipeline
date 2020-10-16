import argparse
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk


DEFAULT_DATA_DIR = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
DEFAULT_PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
DEFAULT_OUTPUT_DIR = "../hecktor_meta"
DEFAULT_MODALITY = "PET"


# Histogram related constants
DEFAULT_QUANTILES_CUTOFF_SUV = (0.01, 0.99)
STANDARD_RANGE_SUV = (0, 20)
DEFAULT_QUANTILES_CUTOFF_HU = (0.01, 0.999)
STANDARD_RANGE_HU = (-1000, 3000)


class StandardHistogramTrainer():
    def __init__(self, modality='PET', quantiles_cutoff=None, images_paths=None):
        self.modality = modality

        if self.modality == 'PET':
            self.quantiles_cutoff = DEFAULT_QUANTILES_CUTOFF_SUV
            self.standard_scale = STANDARD_RANGE_SUV
        elif self.modality == 'CT':
            self.quantiles_cutoff = DEFAULT_QUANTILES_CUTOFF_HU
            self.standard_scale = STANDARD_RANGE_HU

        self.images_paths = images_paths

    def train(self):
        percentiles_cutoff = 100 * np.array(self.quantiles_cutoff)
        percentiles_database = []
        percentiles = self._get_percentiles(percentiles_cutoff)

        for image_file_path in tqdm(self.images_paths):
            image_sitk = sitk.ReadImage(image_file_path)
            image_np = sitk.GetArrayFromImage(image_sitk)

            if self.modality == 'CT': # If CT is given, set all out-of-bound region values equal to air HU (-1000)
                image_np = np.clip(image_np, -1000, image_np.max())

            percentile_values = np.percentile(image_np, percentiles)
            percentiles_database.append(percentile_values)

        percentiles_database = np.vstack(percentiles_database)
        mapping = self._get_average_mapping(percentiles_database)
        return mapping

    def _get_percentiles(self, percentiles_cutoff):
        quartiles = np.arange(25, 100, 25).tolist()
        deciles = np.arange(10, 100, 10).tolist()
        all_percentiles = list(percentiles_cutoff) + quartiles + deciles
        percentiles = sorted(set(all_percentiles))
        return np.array(percentiles)

    def _get_average_mapping(self, percentiles_database):
        pc1 = percentiles_database[:, 0]
        pc2 = percentiles_database[:, -1]
        s1, s2 = self.standard_scale
        slopes = (s2 - s1) / (pc2 - pc1)
        slopes = np.nan_to_num(slopes)
        intercepts = np.mean(s1 - slopes * pc1)
        num_images = len(percentiles_database)
        final_map = slopes.dot(percentiles_database) / num_images + intercepts
        return final_map



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

    with open(args.patient_id_file, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split("\n") if p_id != ""]

    if args.modality == 'PET':
        patient_PET_paths = [f"{args.data_dir}/{p_id}_pt.nii.gz" for p_id in patient_ids]
        histogram_trainer = StandardHistogramTrainer(modality='PET', images_paths=patient_PET_paths)
        PET_landmarks = histogram_trainer.train()
        np.save(f"{args.output_dir}/hist_landmarks_PET.npy", PET_landmarks)

    elif args.modality == 'CT':
        patient_CT_paths = [f"{args.data_dir}/{p_id}_ct.nii.gz" for p_id in patient_ids]
        histogram_trainer = StandardHistogramTrainer(modality='CT', images_paths=patient_CT_paths)
        CT_landmarks = histogram_trainer.train()
        np.save(f"{args.output_dir}/hist_landmarks_CT.npy", CT_landmarks)


if __name__ == '__main__':
    args = get_args()
    main(args)
