'''This script is used to prepare the .csv file of the Skullbreaker dataset.
Each complete skull will have five rows in the .csv file, each row will have the path of the complete skull and the path of the skull with a defect.
The defects are: bilateral, frontoorbital, parietotemporal, random_1, random_2.
The training and eval split will be done 80% and 20% respectively.
'''
import os
import numpy as np
import csv
import nrrd
import torch
import torch.nn as nn
import torch.utils.data
import nibabel
import pandas as pd


def prepare_csv_skullbreaker(main_path):
    # Write .csv based on the main path and the folder "complete_skull".

    # Get the list of all the files in the folder "complete_skull"
    complete_skull_path = os.path.join(main_path, "complete_skull")
    complete_skull_files = os.listdir(complete_skull_path)
    bilateral_skull_path = os.path.join(main_path, "defective_skull", "bilateral")
    frontoorbital_skull_path = os.path.join(main_path, "defective_skull", "frontoorbital")
    parietemporal_skull_path = os.path.join(main_path, "defective_skull", "parietotemporal")
    random_1_skull_path = os.path.join(main_path, "defective_skull", "random_1")
    random_2_skull_path = os.path.join(main_path, "defective_skull", "random_2")

    # Write the .csv file
    with open(os.path.join(main_path, "bc_wdm_skullbreak.csv"), "w") as f:
        # Write the header
        f.write("image_path, missing \n")
        for file in complete_skull_files:
            f.write(f"{os.path.join(complete_skull_path, file)},{os.path.join(bilateral_skull_path, file)}\n")
            f.write(f"{os.path.join(complete_skull_path, file)},{os.path.join(frontoorbital_skull_path, file)}\n")
            f.write(f"{os.path.join(complete_skull_path, file)},{os.path.join(parietemporal_skull_path, file)}\n")
            f.write(f"{os.path.join(complete_skull_path, file)},{os.path.join(random_1_skull_path, file)}\n")
            f.write(f"{os.path.join(complete_skull_path, file)},{os.path.join(random_2_skull_path, file)}\n")

    return None


def add_set_train_or_test_to_csv(main_path):
    '''
    This function will add a column to the csv file with the set of the image. Train or test set.
    There are two .csv that exist and contain the patients used for training and the ones used for test.
    The patients in these files are read and the mode is added to the main .csv file.
    '''

    # Read the patients in the train and the test .csv
    train_patients = np.loadtxt(os.path.join(main_path, "skullbreak_train_split.csv"), dtype=str)
    test_patients = np.loadtxt(os.path.join(main_path, "skullbreak_test_split.csv"), dtype=str)

    print(train_patients)
    print(test_patients)

    # Read the main .csv file and add a third column to every patient
    with open(os.path.join(main_path, "bc_wdm_skullbreak.csv"), "r") as infile, \
            open(os.path.join(main_path, "bc_wdm_skullbreak_correct_split.csv"), "w", newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        writer.writerow(["image_path", "missing", "mode"])

        # Skip the header of the input file
        next(reader)

        for row in reader:
            image_path, missing = row
            image_name = image_path.split("/")[-1]

            if image_name in train_patients:
                print(f'{image_name}, {missing}, train')
                writer.writerow([image_path, missing, "train"])
            elif image_name in test_patients:
                print(f'{image_path}, {missing}, test')
                writer.writerow([image_path, missing, "test"])
            else:
                print(f"Patient {image_name} not found in train or test set.")

def dataset_to_nifti(new_dataset_path, final_image_size=512):
    '''
    This function transforms the .nrrd files in the dataset to the .nii files
    It creates three different versions of the dataset: 512, 256 and 128 resolution.
    :param new_dataset_path: The path to the new dataset
    '''

    # Create the new dataset path
    os.makedirs(new_dataset_path, exist_ok=True)

    # Create the subfolders in the new dataset path and add them to a list
    image_folders = []
    # complete_skull_path = os.path.join(new_dataset_path, "complete_skull")
    # image_folders.append(complete_skull_path)
    # os.makedirs(complete_skull_path, exist_ok=True)
    #
    # defective_skull_bilateral_path = os.path.join(new_dataset_path, "defective_skull", "bilateral")
    # image_folders.append(defective_skull_bilateral_path)
    # os.makedirs(defective_skull_bilateral_path, exist_ok=True)
    #
    # defective_skull_frontoorbital_path = os.path.join(new_dataset_path, "defective_skull", "frontoorbital")
    # image_folders.append(defective_skull_frontoorbital_path)
    # os.makedirs(defective_skull_frontoorbital_path, exist_ok=True)
    #
    # defective_skull_parietotemporal_path = os.path.join(new_dataset_path, "defective_skull", "parietotemporal")
    # image_folders.append(defective_skull_parietotemporal_path)
    # os.makedirs(defective_skull_parietotemporal_path, exist_ok=True)
    #
    # defective_skull_random_1_path = os.path.join(new_dataset_path, "defective_skull", "random_1")
    # image_folders.append(defective_skull_random_1_path)
    # os.makedirs(defective_skull_random_1_path, exist_ok=True)
    #
    # defective_skull_random_2_path = os.path.join(new_dataset_path, "defective_skull", "random_2")
    # image_folders.append(defective_skull_random_2_path)
    # os.makedirs(defective_skull_random_2_path, exist_ok=True)
    #
    # implant_bilateral_path = os.path.join(new_dataset_path, "implant", "bilateral")
    # image_folders.append(implant_bilateral_path)
    # os.makedirs(implant_bilateral_path, exist_ok=True)

    implant_frontoorbital_path = os.path.join(new_dataset_path, "implant", "frontoorbital")
    image_folders.append(implant_frontoorbital_path)
    os.makedirs(implant_frontoorbital_path, exist_ok=True)

    # implant_parietotemporal_path = os.path.join(new_dataset_path, "implant", "parietotemporal")
    # image_folders.append(implant_parietotemporal_path)
    # os.makedirs(implant_parietotemporal_path, exist_ok=True)
    #
    # implant_random_1_path = os.path.join(new_dataset_path, "implant", "random_1")
    # image_folders.append(implant_random_1_path)
    # os.makedirs(implant_random_1_path, exist_ok=True)
    #
    # implant_random_2_path = os.path.join(new_dataset_path, "implant", "random_2")
    # image_folders.append(implant_random_2_path)
    # os.makedirs(implant_random_2_path, exist_ok=True)

    # For each image_folder, copy the 3d structure from a .nrrd files to .nii.gz files
    for image_folder in image_folders:
        for file in os.listdir(image_folder):
            nrrd_image_path = os.path.join(image_folder, file)
            nii_image_path = os.path.join(image_folder, file.split(".")[0] + ".nii.gz")

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


            # Save the image as a .nii.gz file
            nibabel.save(nibabel.Nifti1Image(image, np.eye(4)), nii_image_path)
            print(f"Saved image to {nii_image_path}")
            # Delete the .nrrd file
            os.remove(nrrd_image_path)

def change_image_paths_in_csv(csv_path, new_path):
    # Read the main .csv as a pandas dataframe
    df = pd.read_csv(csv_path)
    # Change the image paths in the first and second column except the header
    # if complete_skull is in the path, change it to new_path/complete_skull/image_name
    # if defective_skull is in the path, change it to new_path/defective_skull/defect_name/image_name
    for index, row in df.iterrows():
        image_path = row["image_path"]
        missing_path = row["missing"]
        image_name = image_path.split("/")[-1]
        if "complete_skull" in image_path:
            df.at[index, "image_path"] = os.path.join(new_path, "complete_skull", image_name)
        else:
            defect_name = missing_path.split("/")[-2]
            df.at[index, "image_path"] = os.path.join(new_path, "defective_skull", defect_name, image_name)
            df.at[index, "missing"] = os.path.join(new_path, "defective_skull", defect_name, image_name)

    # Save the new .csv
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    #main_path = "Documents/GitHub/bc-wdm-3d"
    #prepare_csv_skullbreaker(main_path)
    #add_set_train_or_test_to_csv(main_path)
    # main_path = "data/Skullbreak_512_nii_gz"
    # dataset_to_nifti(main_path, final_image_size=512)
    # main_path = "data/Skullbreak_256_nii_gz"
    # dataset_to_nifti(main_path, final_image_size=256)
    main_path = "data/SkullBreak_128_nii_gz"
    dataset_to_nifti(main_path, final_image_size=128)
    print("Done!")
