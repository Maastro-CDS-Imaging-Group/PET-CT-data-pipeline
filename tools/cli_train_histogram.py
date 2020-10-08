import numpy as np
from torchio.transforms import HistogramStandardization



DATA_DIR = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
PATIENT_ID_FILE = "../hecktor_meta/patient_IDs_train.txt"
OUTPUT_FILE_PET = "../hecktor_meta/hist_landmarks_PET.txt"
OUTPUT_FILE_CT = "../hecktor_meta/hist_landmarks_CT.txt"


with open(PATIENT_ID_FILE, 'r') as pf:
    patient_ids = [p_id for p_id in pf.read().split("\n") if p_id != ""]


# PET
patient_PET_paths = [f"{DATA_DIR}/{p_id}_pt.nii.gz" for p_id in patient_ids]
PET_landmarks = HistogramStandardization.train(patient_PET_paths, cutoffs=(0, 25))
np.save(OUTPUT_FILE_PET, PET_landmarks)

# CT
patient_CT_paths = [f"{DATA_DIR}/{p_id}_ct.nii.gz" for p_id in patient_ids]
CT_landmarks = HistogramStandardization.train(patient_CT_paths)
np.save(OUTPUT_FILE_CT, CT_landmarks)
