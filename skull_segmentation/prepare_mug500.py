''' This script is used to obtain a clean mask of the skull from the MUG5000+ dataset.
This is done by cropping the bottom part of the image, just bellow the ocular cavities, and then using
morpholoid operations to obtain a clean mask of the skull.
This script was abadoned.'''

import SimpleITK as sitk
import os
import numpy as np
import csv
import nrrd
import torch
import torch.nn as nn
import torch.utils.data
import nibabel
import pandas as pd
import scipy.ndimage as ndimage



def nrrd_to_nitfi(nrrd_path, nitfi_path, final_image_size=256):
    """
    Convert a nrrd file to a nitfi file
    :param nrrd_path: The path to the nrrd file
    :param nitfi_path: The path to the nitfi file
    """
    # For each image_folder, copy the 3d structure from a .nrrd files to .nii.gz files

    for file in os.listdir(nrrd_path):
        nrrd_image_path = os.path.join(nrrd_path, file)
        nii_image_path = os.path.join(nitfi_path, file.split(".")[0] + ".nii.gz")

        # Check if the file nii_image_path already exists
        if os.path.exists(nii_image_path):
            print(f"File {nii_image_path} already exists. Skipping.")
            continue

        print(f"Reading {nrrd_image_path}")
        # Read the .nrrd file
        image, _ = nrrd.read(nrrd_image_path)
        print(f"Image shape: {image.shape}")

        if final_image_size == 512:
            image = image
        elif final_image_size == 256:
            # Downsamples the image to 256x256x256 by averaging the values but after binarizing the image
            downsample = nn.AvgPool3d(kernel_size=2, stride=2)
            image = downsample(torch.Tensor(image).unsqueeze(dim=0))
        elif final_image_size == 128:
            # Downsamples the image to 128x128x128 by averaging the values but after binarizing the image
            downsample = nn.AvgPool3d(kernel_size=4, stride=4)
            image = downsample(torch.Tensor(image).unsqueeze(dim=0))
        else:
            raise ValueError("The final_image_size must be 512, 256 or 128.")

        # Binarizes the image
        image[image >= 0.5] = 1
        image[image < 0.5] = 0

        # Turn the tensor image into a numpy array if it is not 512x512x512
        if final_image_size != 512:
            image = image.squeeze().numpy()
            print(f"New image shape: {image.shape}")
        else:
            image = image.squeeze()
            print(f"New image shape: {image.shape}")

        # Fill small holes
        image = ndimage.binary_closing(image, structure=np.ones((3, 3, 3))).astype(np.uint8)

        # Remove small objects
        labeled_image, num_features = ndimage.label(image)
        sizes = ndimage.sum(image, labeled_image, range(num_features + 1))
        mask_size = sizes < 2000  # Adjust the size threshold as needed
        remove_pixel = mask_size[labeled_image]
        labeled_image[remove_pixel] = 0
        image = labeled_image > 0
        image = image.astype(np.float32)

        # Save the image as a .nii.gz file
        nibabel.save(nibabel.Nifti1Image(image, np.eye(4)), nii_image_path)
        print(f"Saved image to {nii_image_path}")

if __name__ == "__main__":
    nrrd_path = 'skull_segmentation/skull_segmentation_test_data'
    nitfi_path = 'skull_segmentation/skull_segmentation_test_data'

    nrrd_to_nitfi(nrrd_path, nitfi_path, final_image_size=256)