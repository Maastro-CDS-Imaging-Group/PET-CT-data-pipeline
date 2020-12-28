import argparse
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk


DEFAULT_DATA_DIR = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFHN_rs113_hecktor_nii"
DEFAULT_PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
DEFAULT_OUTPUT_DIR = "../hecktor_meta/full_head_neck_crop"
DEFAULT_MODALITY = "PET"
DEFAULT_CROSSVAL_CENTRE = "CHUM"

# Histogram related constants
DEFAULT_QUANTILES_CUTOFF_SUV = (0, 0.999)
DEFAULT_QUANTILES_CUTOFF_HU = (0, 0.999)

STANDARD_INTENSITY_RANGE = (0, 1)


class StandardHistogramTrainer():
    def __init__(self, modality='PET', quantiles_cutoff=None, images_paths=None):
        self.standard_scale = STANDARD_INTENSITY_RANGE

        self.modality = modality

        if self.modality == 'PET':
            self.quantiles_cutoff = DEFAULT_QUANTILES_CUTOFF_SUV
        elif self.modality == 'CT':
            self.quantiles_cutoff = DEFAULT_QUANTILES_CUTOFF_HU

        self.images_paths = images_paths

    def train(self):
        percentiles_cutoff = 100 * np.array(self.quantiles_cutoff)
        percentiles_database = []
        percentiles = self._get_percentiles(percentiles_cutoff)

        for image_file_path in tqdm(self.images_paths):
            image_sitk = sitk.ReadImage(image_file_path)
            image_np = sitk.GetArrayFromImage(image_sitk)

            # If CT, then make the background HU same as air HU (i.e. -1000)
            if self.modality == 'CT':
                image_np = np.clip(image_np, -1000, image_np.max())

            # If PET, then make invalid (negative) SUVs equal to 0
            if self.modality == 'PET':
                image_np = np.clip(image_np, 0, image_np.max())


            # Get percentile values and store them
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
                        default=DEFAULT_MODALITY,
                        help="'PET' or 'CT'"
                        )
    parser.add_argument("--crossval_centre",
                        type=str,
                        required=True,
                        default=DEFAULT_CROSSVAL_CENTRE,
                        help="CHGJ, CHMR, CHUM, CHUS, None"
                        )

    args = parser.parse_args()
    return args


def main(args):

    print("Modality:", args.modality)
    print("Crossval split:", args.crossval_centre)

    with open(args.patient_id_file, 'r') as pf:
        patient_ids = [p_id for p_id in pf.read().split("\n") if p_id != ""]

    if args.crossval_centre != "None":
        patient_ids = [p_id for p_id in patient_ids if args.crossval_centre not in p_id]

    if args.modality == 'PET':
        patient_PET_paths = [f"{args.data_dir}/{p_id}_pt.nii.gz" for p_id in patient_ids]
        histogram_trainer = StandardHistogramTrainer(modality='PET', images_paths=patient_PET_paths)
        PET_landmarks = histogram_trainer.train()
        np.savetxt(f"{args.output_dir}/crossval_{args.crossval_centre}-histogram_landmarks_PET.txt", PET_landmarks)

    elif args.modality == 'CT':
        patient_CT_paths = [f"{args.data_dir}/{p_id}_ct.nii.gz" for p_id in patient_ids]
        histogram_trainer = StandardHistogramTrainer(modality='CT', images_paths=patient_CT_paths)
        CT_landmarks = histogram_trainer.train()
        np.savetxt(f"{args.output_dir}/crossval_{args.crossval_centre}-histogram_landmarks_CT.txt", CT_landmarks)


if __name__ == '__main__':
    args = get_args()
    main(args)
