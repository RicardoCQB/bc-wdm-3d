'''
This script is used to predict several bone structures using the TotalSegmentator models.
'''

import os
import subprocess

input_directory = "data/clinical_cases/nifti/"
output_directory = "data/clinical_cases/nifti/"
roi_subset = "skull"

# Loop through the files in the input directory
for patient_folder in os.listdir(input_directory):
    patient_folder_path = os.path.join(input_directory, patient_folder)
    patient_file_path = os.path.join(patient_folder_path, patient_folder + ".nii.gz")

    for roi in roi_subset.split(" "):
        # Check if the roi .nii.gz file exists
        roi_file_path = os.path.join(patient_folder_path, f'{roi}.nii.gz')
        if os.path.exists(roi_file_path):
            print(f'ROI file already exists: {roi_file_path}')
        else:
            # Build the command
            #command = f"TotalSegmentator -i '{patient_file_path}' -o '{patient_output_directory}' --roi_subset '{roi}' --ta total -ml"
            command = f'TotalSegmentator -i {patient_file_path} -o {patient_folder_path} --roi_subset {roi} --device gpu:0'
            print("Running:", command)
            # Execute the command
            subprocess.run(command, shell=True, check=True)


