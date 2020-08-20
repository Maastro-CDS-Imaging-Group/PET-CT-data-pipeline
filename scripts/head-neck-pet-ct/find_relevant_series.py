"""
Input: Dataset dir path

Output: JSON file containing series paths and other info for every patient 

	- Output dict structure -- {PatientID : {PT: [ {Path : <path>,
												   StudyInstanceUID : <study instance UID>,
												   Spacing : <(pxwidth, pxheight, pxdepth)>,
												   Size : <(width, height, depth)>
												   Physical size : <spacing * size>}], 

											 CT: [ {Path : <path>,
											        StudyInstanceUID : <study instance UID>,
												   Spacing : <(pxwidth, pxheight, pxdepth)>,
												   Size : <(width, height, depth)>
												   Physical size : <spacing * size>},

												   {Path : <path>,
												   StudyInstanceUID : <study instance UID>,
												   Spacing : <(pxwidth, pxheight, pxdepth)>,
												   Size : <(width, height, depth)>
												   Physical size : <spacing * size>} ],

											 RTSTRUCT over PT: {Path : <path>,
											                    StudyInstanceUID : <study instance UID>,
												                StructureSetName: <indication of ct or pt>},

											 RTSTRUCT over CT: {Path : <path>,
											                    StudyInstanceUID : <study instance UID>,
												                StructureSetName : <indication of ct or pt>}
											 }
								}
"""

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

    parser.add_argument("--output_file",
                        type=str,
                        default="./find_relevant_series output/paths_to_relevant_series.json",
                        help="Path to the output file")

    args = parser.parse_args()
    return args


def get_series_dirs(patient_dir):
    patient_subdirs = sorted(os.listdir(patient_dir))
    series_dirs = []
    for subdir in patient_subdirs:
        series_dirs.extend( [ patient_dir+'/'+subdir+'/'+s+'/' for s in sorted(os.listdir(patient_dir+'/'+subdir)) ] )
    return series_dirs


def main(args):

	# All patient IDs are checked
	# patient_ids = sorted(os.listdir(args.data_dir))

	# Only the "outlier subjects" are checked
	with open("./find_relevant_studies output/outlier_subjects.json", 'r') as osf:
		outlier_subjects_dict = json.loads(osf.read())
	patient_ids = outlier_subjects_dict.keys()

	output_dict = {} # Main output dictionary

	#sitk_series_reader = sitk.ImageSeriesReader()

	for patient_id in patient_ids:
		print("\nProcessing patient:", patient_id)
		
		# Populate this dict with data from *all* CT, PT and RTSTRUCT series, then filter it in steps
		patient_dict = { 'PT' : [],
						 'CT' : [],
						 'RTSTRUCT' : []} 

		patient_dir = args.data_dir + '/' + patient_id
		series_dirs = get_series_dirs(patient_dir)

		print("Total series found:", len(series_dirs))

		for series_dir in series_dirs:
			sample_dcm = sorted(os.listdir(series_dir))[0]
			dcm_data = pydicom.dcmread(series_dir + '/' + sample_dcm)
				
			# Apply Filter 1 while populating the patient dictionary --
			if dcm_data.Modality in ['PT', 'CT']:
				# dcm_file_paths = sitk_series_reader.GetGDCMSeriesFileNames(series_dir)
				# sitk_image = sitk.ReadImage(dcm_file_paths)
				# pixel_spacing = sitk_image.GetSpacing()
				# image_size = sitk_image.GetSize()
				# physical_size = [image_size[i] * pixel_spacing[i] for i in range(3)]
				# patient_dict[dcm_data.Modality].append( {'Path' : series_dir,
				#                                          'StudyInstanceUID' : dcm_data.StudyInstanceUID,
				# 										   'Spacing' : pixel_spacing,
				# 										   'Size' : image_size,
				# 			                        	   'Physical size' : physical_size
				# 										}
				#                                       )	

				patient_dict[dcm_data.Modality].append( {'Path' : series_dir,
				                                         'StudyInstanceUID' : dcm_data.StudyInstanceUID
				                                        }
			                                          )	

			elif dcm_data.Modality == 'RTSTRUCT':
				if dcm_data.StructureSetName in ['RTstruct_CTsim->PET(PET-CT)', 'RTstruct_CTsim->CT(PET-CT)']:
					patient_dict['RTSTRUCT'].append( {'Path' : series_dir,
					                                  'StudyInstanceUID' : dcm_data.StudyInstanceUID,
													  'StructureSetName' : dcm_data.StructureSetName
													 }
					                               ) 
			
		# Filter the patient dictionary for multiple CTs --
		if len(patient_dict['CT']) > 1:
			rtstruct_uids = [rtstruct_info['StudyInstanceUID'] for rtstruct_info in patient_dict['RTSTRUCT']]
			for i, ct_info in enumerate(patient_dict['CT']):
				ct_uid = ct_info['StudyInstanceUID']
				if ct_uid not in rtstruct_uids: # If Study UID doesn't match to any of the tose of the 2 RTSTRUCTs, remove the CT series
					patient_dict['CT'].pop(i)
		


		# Safety checks and warnings ---

		# 1. Multiple PT series 
		if len(patient_dict['PT']) > 1:
			print("[Warning] Multiple PT series found!")


		print("Relevant series -- ")
		for key in patient_dict.keys():
			print(key, ":", len(patient_dict[key]))

		output_dict[patient_id] = patient_dict	

	with open(args.output_file, 'w') as of:
		output_string = json.dumps(output_dict)
		of.write(output_string)



if __name__ == '__main__':
	args = parse_args()
	main(args)