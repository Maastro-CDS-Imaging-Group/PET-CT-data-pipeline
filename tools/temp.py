import numpy as np
from pathlib import Path
import SimpleITK as sitk


data_dir = Path("../../../Datasets/HECKTOR/hecktor_train/rs113_hecktor_nii")

patient_ids = [str(dir_path).split('/')[-1] \
                   for dir_path in sorted(list(data_dir.glob("*/")))]

print("Total patients found:", len(patient_ids))

print(f"{data_dir}")

for p_id in patient_ids:
    source_patient_dir = f"{data_dir}/{p_id}"

    # # CT
    # ct_image_path = f"{source_patient_dir}/{p_id}_ct.nrrd"
    # ct_sitk = sitk.ReadImage(str(ct_image_path))
    # print(ct_sitk.GetSize())
    # print(ct_sitk.GetSpacing())
    # print()

    # GTV
    gtv_image_path = f"{source_patient_dir}/{p_id}_ct_gtvt.nrrd"
    gtv_sitk = sitk.ReadImage(str(gtv_image_path))
    gtv_np = sitk.GetArrayFromImage(gtv_sitk)
    print(np.unique(gtv_np))