import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import random

def crop_to_skull_region(nifti_path, slice_number, output_path):
    """
    Crop the full-body CT to the skull region above the given slice number.
    :param nifti_path: Path to the input Nifti file.
    :param slice_number: Slice number to crop above.
    :param output_path: Path to save the cropped Nifti file.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    cropped_data = data[:, :, slice_number:]
    cropped_img = nib.Nifti1Image(cropped_data, img.affine)
    nib.save(cropped_img, output_path)
    print(f"Cropped image saved to {output_path}")

def segment_bone_tissue(nifti_path, hu_threshold, output_path):
    """
    Segment the bone tissue using HU thresholding.
    :param nifti_path: Path to the input Nifti file.
    :param hu_threshold: HU threshold for bone segmentation.
    :param output_path: Path to save the binary skull mask.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    bone_mask = (data > hu_threshold).astype(np.uint8)
    bone_img = nib.Nifti1Image(bone_mask, img.affine)
    nib.save(bone_img, output_path)
    print(f"Binary skull mask saved to {output_path}")



if __name__ == "__main__":
    # Example usage
    nifti_path = 'path_to_fullbody_ct.nii.gz'
    cropped_output_path = 'path_to_cropped_skull.nii.gz'
    binary_skull_output_path = 'path_to_binary_skull_mask.nii.gz'
    defect_output_path = 'path_to_skull_with_defects.nii.gz'

    # Crop to skull region
    crop_to_skull_region(nifti_path, slice_number=100, output_path=cropped_output_path)

    # Segment bone tissue
    segment_bone_tissue(cropped_output_path, hu_threshold=300, output_path=binary_skull_output_path)
