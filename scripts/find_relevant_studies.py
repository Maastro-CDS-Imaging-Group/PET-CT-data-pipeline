import os, argparse, json

import numpy as np

import pydicom
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        type=str, 
                        default="../Data/Head-Neck-PET-CT", 
                        help="Path to the Head-Neck-PET-CT data")
    
    parser.add_argument("--output_file_1", 
                        type=str, 
                        default="./paths_to_relevant_studies.txt", 
                        help="Path to the primary ouput file to store the paths to relevant studies")


    parser.add_argument("--output_file_2", 
                        type=str, 
                        default="./outlier_subjects.json", 
                        help="Path to the ouput file to store outlier subjects info - i.e. subjects with separate PET and CT studies")

    
    args = parser.parse_args()
    return args


def main(args):

    subject_IDs = sorted(os.listdir(args.data_dir))

    # For each subject, identify the studies that contain both PET and CT series

    outlier_subjects_dict = {} # Subjects for whom PET and CT series do not exist within the same study
    

    for subject in subject_IDs:
        print("Checking subject: ", subject)

        subject_dir = args.data_dir + '/' + subject + "/"
        studies = sorted(os.listdir(subject_dir))

        found_relevant_study = False
        all_study_modalities = [] 
        for study in studies:
            study_dir = subject_dir + study + "/"
            list_of_series = sorted(os.listdir(study_dir))

            study_modalities = []

            for series_name in list_of_series:
                series_dir = study_dir + series_name + "/"
                sample_dcm_file_name = sorted(os.listdir(series_dir))[0]
                dcm_object = pydicom.dcmread(series_dir + sample_dcm_file_name, stop_before_pixels=True)
                study_modalities.append(dcm_object.Modality)

            print(study_modalities)
            all_study_modalities.append(tuple(study_modalities))

            if ("PT" in study_modalities and "CT" in study_modalities):
                found_relevant_study = True
                with open(args.output_file_1, 'a') as rel_studies_file:
                    rel_studies_file.write(study_dir + "\n")

        if not found_relevant_study: # Remember the outlier subjects
            outlier_subjects_dict[subject] = all_study_modalities
            print("Outlier subject!")
            print("All study modalites:", outlier_subjects_dict[subject])

    # Store the outlier subjects info in the file. Need to manually deal with them.
    outlier_subjects_info = json.dumps(outlier_subjects_dict)
    with open(args.output_file_2, 'w') as outlier_subjects_file:
        outlier_subjects_file.write(outlier_subjects_info)


if __name__ == '__main__':
    args = parse_args()
    main(args)