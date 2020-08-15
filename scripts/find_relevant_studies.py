import os, argparse

import numpy as np

import pydicom
import SimpleITK as sitk


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        type=str, 
                        default="../Data/Head-Neck-PET-CT", 
                        help="Path to the Head-Neck-PET-CT data")
    
    parser.add_argument("--output_file", 
                        type=str, 
                        default="./paths_to_relevant_studies.txt", 
                        help="Path to the ouput file")
    
    args = parser.parse_args()
    return args


def main(args):

    subject_IDs = sorted(os.listdir(args.data_dir))

    # For each subject, identify the studies that contain both PET and CT series

    subjects_with_separate_studies = [] # Subjects for whom PET and CT series do not exist within the same study

    for subject in subject_IDs:
        print("Checking subject: ", subject)

        subject_dir = args.data_dir + '/' + subject + "/"
        studies = sorted(os.listdir(subject_dir))

        found_relevant_study = False
        for study in studies:
            study_dir = subject_dir + study + "/"
            list_of_series = sorted(os.listdir(study_dir))

            available_modalities = []

            for series_name in list_of_series:
                series_dir = study_dir + series_name + "/"
                sample_dcm_file_name = sorted(os.listdir(series_dir))[0]
                dcm_object = pydicom.dcmread(series_dir + sample_dcm_file_name, stop_before_pixels=True)
                available_modalities.append(dcm_object.Modality)

            print(available_modalities)
            if ("PT" in available_modalities and "CT" in available_modalities):
                found_relevant_study = True
                with open(args.output_file, 'a') as out_file:
                    out_file.write(study_dir + "\n")

        if not found_relevant_study: # Remember the special cases
            subjects_with_separate_studies.append(subject)

    # Display the special cases and note them in the file. Need to manually deal with them.
    if len(subjects_with_separate_studies) > 0:
        message = "\nSubjects with separate PET and CT studies:" + ' '.join(subjects_with_separate_studies)
        print(message)
        with open(args.output_file, 'a') as out_file:
            out_file.write(message)


if __name__ == '__main__':
    args = parse_args()
    main(args)