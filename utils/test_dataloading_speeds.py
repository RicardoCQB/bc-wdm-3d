'''
This script is used to test the speed of the dataloading strategies: loading the images from a .nii.gz file, and loading the images from a .npy file.
'''

import os
import nibabel
import numpy as np
import torch
import time
import nrrd

def load_nii_gz(image_pth, condition_pth):
    image = nibabel.load(image_pth).get_fdata()
    condition_image = nibabel.load(condition_pth).get_fdata()
    return image, condition_image


def load_npy(image_pth, condition_pth):
    image = np.load(image_pth)
    condition_image = np.load(condition_pth)
    return image, condition_image

def nifti_to_npy(nifti_pth, npy_pth):
    image = nibabel.load(nifti_pth).get_fdata()
    np.save(npy_pth, image)


def test_speeds(image_pth, condition_pth, image_npy_pth, condition_npy_pth, image_nrrd_pth, condition_nrrd_pth):

    # Load the .nii.gz files
    start = time.time()
    image, condition_image = load_nii_gz(image_pth, condition_pth)
    # Transform the images to torch tensors
    image = torch.Tensor(image)
    condition_image = torch.Tensor(condition_image)
    nii_load_time = time.time() - start
    print(f"Time to load .nii.gz files and turn them into tensors: {nii_load_time}")

    # Load the .npy files
    start = time.time()
    image, condition_image = load_npy(image_npy_pth, condition_npy_pth)
    # Transform the images to torch tensors
    image = torch.Tensor(image)
    condition_image = torch.Tensor(condition_image)
    npy_load_time = time.time() - start
    print(f"Time to load .npy files and turn them into tensors: {npy_load_time}")

    # Load the .nrrd files
    start = time.time()
    image, header = nrrd.read(image_nrrd_pth)
    condition_image, header = nrrd.read(condition_nrrd_pth)
    # Transform the images to torch tensors
    image = torch.Tensor(image)
    condition_image = torch.Tensor(condition_image)
    nrrd_load_time = time.time() - start
    print(f"Time to load .nrrd files and turn them into tensors: {nrrd_load_time}")

    # Calculate how many times faster the .npy files are
    time_saving_factor = nii_load_time / npy_load_time
    print(f".npy files are {time_saving_factor} times faster than .nii.gz files")
    time_save_nrrd = nrrd_load_time / nii_load_time
    print(f".nii.gz files are {time_save_nrrd} times faster than .nrrd files")

    # Calculate how much memory in megabytes is saved by using .npy
    memory_factor_npy_to_nifti = os.path.getsize(image_npy_pth) / os.path.getsize(image_pth)
    print(f'memory factor npy to nii.gz = {memory_factor_npy_to_nifti}')
    print(f'15gb nii.gz = {15*memory_factor_npy_to_nifti} gb npy')
    memory_factor_nrrd_to_nifti = os.path.getsize(image_pth) / os.path.getsize(image_nrrd_pth)
    print(f'memory factor nii.gz to nrrd = {memory_factor_nrrd_to_nifti}')
    print(f'15gb nii.gz = {15*memory_factor_nrrd_to_nifti} gb nrrd')




if __name__ == '__main__':
    image_pth = "utils/000_image.nii.gz"
    condition_pth = "utils/000_condition.nii.gz"
    image_npy_pth = "utils/000_image.npy"
    condition_npy_pth = "utils/000_condition.npy"
    image_nrrd_pth = "utils/000_image.nrrd"
    condition_nrrd_pth = "utils/000_condition.nrrd"

    nifti_to_npy(image_pth, image_npy_pth)
    nifti_to_npy(condition_pth, condition_npy_pth)
    test_speeds(image_pth, condition_pth, image_npy_pth, condition_npy_pth, image_nrrd_pth, condition_nrrd_pth)
