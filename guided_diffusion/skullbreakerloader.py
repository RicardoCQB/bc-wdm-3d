import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import os.path
import nibabel
import pandas as pd
import torchio as tio

class SkullBreakDataset(torch.utils.data.Dataset):
    '''
    This dataloader will load the skullbreaker dataset from the given path. The main image will be the skull with no defects
    and the image that will be called condition_image will be the skull with defects. There must be a .csv file witht he path
    of each image and the path of the condition image. The csv file must have the following structure:
    path, condition_path
    .../path1.nrrd, .../condition_path1.nrrd

    The defects are: bilateral, frontoorbital, parietotemporal, random_1, random_2 and each image will have 5 defects.
    And each time the dataloader is called, it will return a random image with a random defect.
    The dataloader will return a tuple with the main image and the condition image.

    '''
    def __init__(self, path, mode="train", augment=False, normalize=None, downsample_factor=4):
        super().__init__()
        self.directory = path
        self.database = []
        self.mode = mode
        self.augment = augment
        self.normalize = normalize or (lambda x: x)
        self.downsample_factor = downsample_factor

        #Open the .csv file as numpy array
        self.database = pd.read_csv(os.path.join(self.directory, 'bc_wdm_skullbreak_cq3defects_test.csv'))
        #self.database = pd.read_csv(os.path.join(self.directory, 'clinical_data.csv'))
        #Turn the pandas dataframe into a numpy array
        self.database = self.database.to_numpy()

        if self.mode == 'train':
            self.database = self.database[self.database[:,2] == 'train']
            print(f'Training with {len(self.database)} samples')
        elif self.mode == 'test':
            self.database = self.database[self.database[:,2] == 'test']
            print(f'Evaluating with {len(self.database)} samples')
        elif self.mode == 'clinical':
            self.database = self.database[self.database[:,2] == 'clinical']
            print(f'Testing clinical data with {len(self.database)} samples')

    def __getitem__(self, x):
        filedict = self.database[x]
        image_pth = filedict[0]
        condition_pth = filedict[1]

        try:
            # Check if the files exist
            if not os.path.exists(image_pth) or not os.path.exists(condition_pth):
                raise FileNotFoundError(f"File not found: {image_pth} or {condition_pth}")

            # Load the .nii.gz files
            image = nibabel.load(image_pth).get_fdata()
            condition_image = nibabel.load(condition_pth).get_fdata()

            # Get the image name from the image_pth
            image_name = image_pth.split('/')[-1].split('.')[0]

            # Get the condition image name
            condition_name = condition_pth.split('/')[-1].split('.')[0]

            # Get the missing type from the condition_pth
            if self.mode == 'train' or self.mode == 'test':
                missing_type = condition_pth.split('/')[-2]
                if missing_type == "defected_skulls":
                    missing_type = condition_name.split("_")[2]
            elif self.mode == 'clinical':
                missing_type = 'clinical'

            # Transform the images to torch tensors
            image = torch.Tensor(image)
            condition_image = torch.Tensor(condition_image)

            # Add another dimension to the images
            image = image.unsqueeze(0)
            condition_image = condition_image.unsqueeze(0)

            # Normalize the images
            if self.normalize is not None:
                image = self.normalize(image)
                condition_image = self.normalize(condition_image)

            if self.mode == 'train':
                return {'image': image, 'missing': condition_image}
            else:
                return {'image': image, 'missing': condition_image, 'subj': image_name, 'missing_type': missing_type}

        except Exception as e:
            print(f"Error loading data for index {x}: {e}")
            return None  # Return None if there is an error

    def __len__(self):
        return len(self.database)

    def resize_image(self, image, target_size):
        """
        Resizes a 3D image tensor to the target size.

        Parameters:
            image (torch.Tensor): The input image tensor of shape (C, D, H, W).
            target_size (tuple): The target size as (D, H, W).

        Returns:
            torch.Tensor: The resized image tensor.
        """
        # Add batch dimension if not present
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Resize using F.interpolate
        resized_image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=False)

        # Remove batch dimension if added
        if resized_image.shape[0] == 1:
            resized_image = resized_image.squeeze(0)

        return resized_image