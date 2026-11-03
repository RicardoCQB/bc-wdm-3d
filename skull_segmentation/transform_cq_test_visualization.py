import nibabel as nib
import numpy as np
import sys
import os


def save_with_identity_affine(input_path, output_path=None):
    # Read the NIfTI volume
    img = nib.load(input_path)
    data = img.get_fdata()

    # Create a new image with an identity affine
    new_img = nib.Nifti1Image(data, affine=np.eye(4))

    # Define the output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        if ext == '.gz':
            base, _ = os.path.splitext(base)
            ext = '.nii.gz'
        else:
            ext = '.nii'
        output_path = f"{base}_identity_affine{ext}"

    # Save the new volume
    nib.save(new_img, output_path)
    print(f"File saved to: {output_path}")


if __name__ == "__main__":
    input_path = 'data/CT480_defects_2.nii.gz'
    output_path = 'data/CT480_defects_2_affined.nii.gz'
    save_with_identity_affine(input_path, output_path)