import json
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm


data_dir = "/home/chinmay/Datasets/HECKTOR/hecktor_train/crFH_rs113_hecktor_nii"
patient_id_filepath = "../hecktor_meta/patient_IDs_train.txt"

output_file_path = "../hecktor_meta/class_frequencies_crFH_rs113_train.txt"


with open(patient_id_filepath, 'r') as pf:
	patient_IDs = [p_id for p_id in pf.read().split('\n') if p_id != '']


class_frequencies = {0: 0, 1: 0}

for p_id in tqdm(patient_IDs):
	GTV_labelmap_sitk = sitk.ReadImage(f"{data_dir}/{p_id}_ct_gtvt.nii.gz")
	GTV_labelmap_np = sitk.GetArrayFromImage(GTV_labelmap_sitk)
	for class_id in class_frequencies.keys():
		class_frequencies[class_id] += np.sum(np.equal(GTV_labelmap_np, class_id))

print(class_frequencies)

with open(output_file_path, 'w') as of:
	message = {}
	for k, v in class_frequencies.items():
		message[str(k)] = str(v)
	of.write(json.dumps(message))