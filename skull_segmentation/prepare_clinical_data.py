import pandas as pd
import os
import csv
import shutil
import dicom2nifti
import pydicom
import os
import nibabel as nib
import torchio as tio
import numpy as np
import nrrd
def is_valid_dicom(file_path):
    """Check if a file is a valid DICOM file."""
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except Exception:
        return False

def convert_dicom_to_nifti(dicom_input_path, nifti_output_path):
    """
    Converts DICOM files for multiple patients into NIfTI format and saves them in corresponding folders.

    Parameters:
        dicom_input_path (str): Path to the folder containing patient DICOM folders.
        nifti_output_path (str): Path to the folder where NIfTI files will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(nifti_output_path):
        os.makedirs(nifti_output_path)

    # Iterate through each patient folder in the DICOM input path
    for patient_folder in os.listdir(dicom_input_path):
        patient_folder_path = os.path.join(dicom_input_path, patient_folder)

        # Skip if it's not a directory
        if not os.path.isdir(patient_folder_path):
            continue

        # Create a corresponding folder in the NIfTI output path
        patient_nifti_folder = os.path.join(nifti_output_path, patient_folder)
        if not os.path.exists(patient_nifti_folder):
            os.makedirs(patient_nifti_folder)

        try:
            # Filter valid DICOM files
            valid_dicom_files = [
                os.path.join(patient_folder_path, f)
                for f in os.listdir(patient_folder_path)
                if is_valid_dicom(os.path.join(patient_folder_path, f))
            ]

            if not valid_dicom_files:
                print(f"No valid DICOM files found in '{patient_folder_path}'. Skipping.")
                continue

            # Convert the DICOM folder to NIfTI and save in the patient's NIfTI folder
            dicom2nifti.convert_directory(patient_folder_path, patient_nifti_folder)
            print(f"Successfully converted DICOM for patient '{patient_folder}' to NIfTI.")
        except Exception as e:
            print(f"Failed to convert DICOM for patient '{patient_folder}': {e}")

def downsample_nifti(input_path, output_path, target_size=(256, 256, 256), is_label=True):
    """
    Downsamples a NIfTI image to the target size while maintaining anatomical proportions.

    Parameters:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the downsampled NIfTI file.
        target_size (tuple): Final output shape (default: 256x256x256).
        is_label (bool): If True, use nearest-neighbor interpolation for label maps.
    """
    try:
        # Load image with spacing using TorchIO
        image_class = tio.LabelMap if is_label else tio.ScalarImage
        image = image_class(input_path)

        # Step 1: Resample to isotropic spacing (e.g., 1 mm³)
        image = tio.Resample(1)(image)

        # Step 2: Resize to target shape (with aspect ratio preserved after resampling)
        image = tio.Resize(target_size)(image)

        # Step 3: Save output as NIfTI using nibabel
        output_array = image.data.squeeze(0).numpy()  # Remove channel dim
        nib_img = nib.Nifti1Image(output_array, affine=image.affine)
        nib.save(nib_img, output_path)
        print(f"Downsampled image saved to: {output_path}")

    except Exception as e:
        print(f"Error downsampling NIfTI file '{input_path}': {e}")

def normalize_and_crop_nifti(input_path, output_path, target_shape=(256, 256, 256), is_label=True):
    """
    Resamples a NIfTI image to 1mm isotropic spacing, then center crops/pads to target_shape.

    Parameters:
        input_path (str): Path to the input NIfTI file.
        output_path (str): Path to save the downsampled NIfTI file.
        target_shape (tuple): Final desired shape (default: 256x256x256).
        is_label (bool): If True, use nearest-neighbor interpolation.
    """
    try:
        # Load NIfTI using torchio
        ImageClass = tio.LabelMap if is_label else tio.ScalarImage
        image = ImageClass(input_path)

        # Step 1: Resample to isotropic spacing (1mm³)
        image = tio.Resample((1, 1, 1))(image)

        # Step 2: Pad/Crop to target shape (centered)
        image = tio.CropOrPad(target_shape)(image)

        # print the affine matrix of the image we are reading
        print(f'image affine matrix: {image.affine}')
        # Step 3: Save back as NIfTI
        output_array = image.data.squeeze(0).numpy()
        nib_img = nib.Nifti1Image(output_array, affine=image.affine)
        nib.save(nib_img, output_path)
        print(f"Saved processed volume: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def reset_affine_center(nifti_path, output_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    new_affine = np.eye(4)
    new_affine[:3, 3] = -0.5 * np.array(data.shape)
    nib.save(nib.Nifti1Image(data, new_affine), output_path)


def process_nifti_files(sample_path, target_path, output_dir):
    """
    Reads two NIfTI files (sample and target), performs binary subtraction, and saves the results.

    Parameters:
        sample_path (str): Path to the sample NIfTI file.
        target_path (str): Path to the target NIfTI file.
        output_dir (str): Directory to save the output files.
    """
    try:
        # Load the sample and target NIfTI files
        sample_img = nib.load(sample_path)
        target_img = nib.load(target_path)

        # Get the data arrays
        sample_data = sample_img.get_fdata()
        target_data = target_img.get_fdata()

        # Perform binary subtraction: sample - target
        implant_data = (sample_data - target_data) > 0  # Binary result (True/False)
        implant_data = implant_data.astype(np.uint8)  # Convert to binary (0/1)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the implant binary volume as NIfTI
        implant_nifti_path = os.path.join(output_dir, "implant.nii.gz")
        implant_nifti = nib.Nifti1Image(implant_data, affine=sample_img.affine)
        nib.save(implant_nifti, implant_nifti_path)
        print(f"Saved implant NIfTI to: {implant_nifti_path}")

        # Save the implant binary volume as NRRD
        implant_nrrd_path = os.path.join(output_dir, "implant.nrrd")
        nrrd.write(implant_nrrd_path, implant_data)
        print(f"Saved implant NRRD to: {implant_nrrd_path}")

        # Save the target volume as NRRD
        target_nrrd_path = os.path.join(output_dir, "target.nrrd")
        nrrd.write(target_nrrd_path, target_data)
        print(f"Saved target NRRD to: {target_nrrd_path}")

        # Save the sample volume as NRRD
        sample_nrrd_path = os.path.join(output_dir, "sample.nrrd")
        nrrd.write(sample_nrrd_path, sample_data)
        print(f"Saved sample NRRD to: {sample_nrrd_path}")

    except Exception as e:
        print(f"Error processing NIfTI files: {e}")

# dicom_clinical_path = "data/clinical_cases/dicom/"
# nifti_output_path = "data/clinical_cases/nifti/"
# convert_dicom_to_nifti(dicom_clinical_path, nifti_output_path)

# input_nifti_path = "data/clinical_cases/nifti_binary_not_resampled/"
# output_nifti_path = "data/clinical_cases/nifti_binary/"
#
# for file in os.listdir(input_nifti_path):
#     file_path = os.path.join(input_nifti_path, file)
#     output_file_path = os.path.join(output_nifti_path, file)
#     # normalize_and_crop_nifti(file_path, output_file_path)
#     # reset_affine_center(file_path, output_file_path)
#     normalize_and_crop_nifti(file_path, output_file_path, target_shape=(256, 256, 256), is_label=True)

# predictions_folder = "results/skullbreaker_unet_400000/"
# nrrd_output = "data/clinical_cases/nrrd/"
#
# for patient in os.listdir(predictions_folder):
#     patient_folder = os.path.join(predictions_folder, patient)
#     sample_path = os.path.join(patient_folder, "clinical_sample.nii.gz")
#     target_path = os.path.join(patient_folder, "clinical_target.nii.gz")
#
#     # Create a folder for the patient in the nrrd_output directory if it does not exist
#     patient_output_folder = os.path.join(nrrd_output, patient)
#     os.makedirs(patient_output_folder, exist_ok=True)
#
#     # Process the NIfTI files for the patient
#     process_nifti_files(sample_path, target_path, patient_output_folder)

# Change the affine matrix of the NIfTI files of the defected skulls and the implants to be centered at the origin.
nifti_defected_folder = 'nifti_binary_defected'
nifti_implants_folder = 'nifti_binary_implant'

for file in os.listdir(nifti_defected_folder):
    file_path = os.path.join(nifti_defected_folder, file)
    output_file_name = file.replace('.nii.gz', '_centered.nii.gz')
    output_file_path = os.path.join(nifti_defected_folder, output_file_name)
    reset_affine_center(file_path, output_file_path)

for file in os.listdir(nifti_implants_folder):
    file_path = os.path.join(nifti_implants_folder, file)
    output_file_name = file.replace('.nii.gz', '_centered.nii.gz')
    output_file_path = os.path.join(nifti_implants_folder, output_file_name)
    reset_affine_center(file_path, output_file_path)