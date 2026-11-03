import random
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import os
import shutil
from skimage.morphology import ball, binary_opening, binary_dilation, binary_erosion


def generate_skull_defects_OLD(original_volume, max_num_defects=3, min_radius=10, max_radius=40, deformation_strength=40,
                           erosion_probability=0.2, erosion_iters=3):
    """
    Generates defects in a binary skull volume.

    Parameters:
        volume (numpy.ndarray): 3D binary array (256x256x256) representing the skull.
        max_num_defects (int): Maximum number of defects to generate, it will be randomly chosen.
        min_radius (int): Minimum radius of defect shapes.
        max_radius (int): Maximum radius of defect shapes.
        deformation_strength (int): Strength of elastic deformation.
        erosion_probability (float): Probability of applying erosion to roughen defect edges.
        erosion_iters (int): Number of erosion iterations to roughen edges.

    Returns:
        numpy.ndarray: Volume with defects applied.
    """
    volume = original_volume.copy()  # Store the original skull
    shape = volume.shape
    num_defects = np.random.randint(1, max_num_defects)

    # Get the coordinates of the non-zero voxels in the binary skull
    non_zero_coords = np.argwhere(volume > 0)

    for _ in range(num_defects):
        defect_mask = np.zeros_like(volume)  # Reset defect mask for each defect

        # Randomly select a center from the non-zero voxels
        center_idx = np.random.choice(len(non_zero_coords))
        cz, cx, cy = non_zero_coords[center_idx]

        # Random radius
        radius = np.random.randint(min_radius, max_radius)

        # Create a spherical defect (or ellipsoid with some variation)
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        deform_factor = np.random.uniform(0.7, 1.3, size=3)  # Increase variability
        mask = ((x - cz) / deform_factor[0]) ** 2 + ((y - cx) / deform_factor[1]) ** 2 + (
                    (z - cy) / deform_factor[2]) ** 2 <= radius ** 2

        defect_mask[mask] = 1

        # Elastic deformation
        def elastic_deform(mask, strength=deformation_strength):
            displacement = [
                ndi.gaussian_filter(np.random.randn(*mask.shape) * strength, sigma=8)
                for _ in range(3)
            ]
            coords = np.meshgrid(
                np.arange(mask.shape[0]), np.arange(mask.shape[1]), np.arange(mask.shape[2]), indexing='ij'
            )
            indices = tuple(
                np.clip(coords[i] + displacement[i], 0, mask.shape[i] - 1).astype(np.float32) for i in range(3)
            )
            return ndi.map_coordinates(mask, indices, order=1, mode='reflect')

        defect_mask = elastic_deform(defect_mask) > 0.5  # Binarize after deformation

        # Apply erosion to roughen the defect edges in 20% of the cases
        if random.random() < erosion_probability:
            for _ in range(erosion_iters):
                defect_mask = ndi.binary_erosion(defect_mask, structure=np.ones((3, 3, 3)))

        # Subtract defect mask from the skull volume
        volume[defect_mask > 0] = 0

    # Compute implant volume (difference between original and defected skull)
    implant_volume = (original_volume - volume) > 0

    return volume, implant_volume


def get_defect_center(shape, region, skull_volume):
    """Returns a random center based on the skull region or anywhere in the skull if region is 'random'."""
    #if region == "random":
    non_zero_coords = np.argwhere(skull_volume > 0)
    return tuple(non_zero_coords[np.random.choice(len(non_zero_coords))])

def get_border(mask):
    """Extracts the border of a defect mask by subtracting eroded version from the dilated version."""
    return binary_dilation(mask, ball(1)) ^ binary_erosion(mask, ball(1))

def get_random_border_point(border_mask):
    """Gets a random point from the border mask."""
    border_coords = np.argwhere(border_mask)
    return tuple(border_coords[np.random.choice(len(border_coords))])


def generate_defect_mask(shape, defect_type, region, skull_volume, max_sphere_radius=40, max_rectangle_size=40):
    """Generates different types of defect masks with customizable max sizes and returns their characteristics."""
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cx, cy = get_defect_center(shape, region, skull_volume)
    characteristics = {}

    if defect_type == "sphere":
        radius = np.random.randint(10, max_sphere_radius)
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        mask = ((x - cz) ** 2 + (y - cx) ** 2 + (z - cy) ** 2) <= radius ** 2
        characteristics = {"type": "sphere", "radius": radius, "center_x": cx, "center_y": cy, "center_z": cz}

    elif defect_type == "rectangle":
        size_x, size_y, size_z = np.random.randint(10, max_rectangle_size, size=3)
        mask[max(0, cz - size_z // 2):min(shape[0], cz + size_z // 2),
             max(0, cx - size_x // 2):min(shape[1], cx + size_x // 2),
             max(0, cy - size_y // 2):min(shape[2], cy + size_y // 2)] = 1
        characteristics = {"type": "rectangle", "size_x": size_x, "size_y": size_y, "size_z": size_z, "center_x": cx,
                           "center_y": cy, "center_z": cz}

        # --- Apply Random Perturbation Before Smoothing ---
    #mask = ndi.gaussian_filter(mask.astype(float), sigma=np.random.uniform(0.5, 2)) > 0.5  # Blur before binarization

    # --- Morphological Opening for Smoother Borders ---
    struct_element = ball(np.random.randint(2, 7))  # Sphere structuring element with random radius
    mask = binary_opening(mask, struct_element)

    # Decide how many additional defects to add (randomized)
    num_extra_defects = np.random.randint(1, 4)

    for _ in range(num_extra_defects):
        # Get the border of the current defect
        border_mask = get_border(mask)

        # If no border points exist, stop adding defects
        if not np.any(border_mask):
            break

        # Get a random point on the border
        cz, cx, cy = get_random_border_point(border_mask)

        # Add another defect at this location
        defect_type = random.choice(["sphere", "rectangle"])
        new_mask = np.zeros(shape, dtype=np.uint8)

        if defect_type == "sphere":
            radius = np.random.randint(10, 30)  # Smaller additional defects
            x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
            new_mask = ((x - cz) ** 2 + (y - cx) ** 2 + (z - cy) ** 2) <= radius ** 2

        elif defect_type == "rectangle":
            size_x, size_y, size_z = np.random.randint(10, 30, size=3)
            new_mask[max(0, cz - size_z // 2):min(shape[0], cz + size_z // 2),
            max(0, cx - size_x // 2):min(shape[1], cx + size_x // 2),
            max(0, cy - size_y // 2):min(shape[2], cy + size_y // 2)] = 1

        # Apply smoothing to the new defect before merging
        new_mask = binary_opening(new_mask, struct_element)

        # Merge new defect with the previous mask
        mask = np.logical_or(mask, new_mask)

    return mask, characteristics

def generate_skull_defects(original_volume, max_num_defects=3, deformation_strength=100, max_sphere_radius=40,
                           max_rectangle_size=40):
    """
    Generates defects in a binary skull volume and returns their characteristics.

    Parameters:
        original_volume (numpy.ndarray): 3D binary array representing the skull.
        max_num_defects (int): Maximum number of defects to generate, randomly chosen.
        deformation_strength (int): Strength of elastic deformation.
        max_sphere_radius (int): Maximum radius for spherical defects.
        max_rectangle_size (int): Maximum size for rectangular defects.

    Returns:
        tuple: (Volume with defects, Implant volume, Defect characteristics)
    """
    volume = original_volume.copy()
    shape = volume.shape
    num_defects = np.random.randint(1, max_num_defects+1)
    defect_characteristics = {
        "num_defects": num_defects,
        "defects": []
    }

    for _ in range(num_defects):
        defect_type = random.choice(["sphere", "rectangle"])
        region = "random"
        defect_mask, characteristics = generate_defect_mask(shape, defect_type, region, original_volume, max_sphere_radius,
                                                            max_rectangle_size)

        # Elastic deformation
        # def elastic_deform(mask, strength=deformation_strength):
        #     displacement = [
        #         ndi.gaussian_filter(np.random.randn(*mask.shape) * strength, sigma=8)
        #         for _ in range(3)
        #     ]
        #     coords = np.meshgrid(
        #         np.arange(mask.shape[0]), np.arange(mask.shape[1]), np.arange(mask.shape[2]), indexing='ij'
        #     )
        #     indices = tuple(
        #         np.clip(coords[i] + displacement[i], 0, mask.shape[i] - 1).astype(np.float32) for i in range(3)
        #     )
        #     return ndi.map_coordinates(mask, indices, order=1, mode='reflect')
        #
        # defect_mask = elastic_deform(defect_mask) > 0.5  # Binarize after deformation

        # Subtract defect mask from the skull volume
        volume[defect_mask > 0] = 0

        # Calculate defect volume
        defect_volume = np.sum(defect_mask)
        characteristics["volume"] = defect_volume
        defect_characteristics["defects"].append(characteristics)

    # Compute implant volume
    implant_volume = (original_volume - volume) > 0
    return volume, implant_volume.astype(np.uint8), defect_characteristics


def create_defects_on_cq500_skulls(dataset_path, num_defected_skulls, output_path):
    '''
    Create skull defects on the skulls of the CQ500 dataset.
    Each file represents a skull volume of a different patient and the defects of this skull
    will be saved with the name of the file and the suffix "_defects_{i}.nii.gz" where i is the number of the defect.
    The implant volume will be saved with the name of the file and the suffix "_implant_{i}.nii.gz" where i is the number of the defect.
    There will also be a .csv file recording every defect created for each skull.
    '''

    # For each directory in the dataset path, read the "patientname_...something.nii.gz" file and the "skull.nii.gz" file
    # Move the original CT file to the "ct" folder in the output_path, move the skull file to the "complete_skull" folder in the output_path
    # but with the "patientname_skull.nii.gz" name. Then create the defects and save them in the "defects" folder in the output_path
    # with the name "patientname_defects_{i}.nii.gz" where i is the number of the defect. The implant volume will be saved in the
    # "implant" folder in the output_path with the name "patientname_implant_{i}.nii.gz" where i is the number of the defect.
    # The characteristics of the defects will be saved in a .csv file in the "defects" folder with the name "defects_characteristics".

    for file in os.listdir(dataset_path):
        patient_folder_path = os.path.join(dataset_path, file)
        patient_file_path = os.path.join(patient_folder_path, file + ".gz")
        skull_volume_path = os.path.join(patient_folder_path, "skull.nii.gz")
        patient_name = file.split("_")[0]

        # Move the patient_file_path file to the "ct" folder in the output_path
        ct_path = os.path.join(output_path, 'ct', f'{patient_name}.nii.gz')
        shutil.copy(patient_file_path, ct_path)

        # Move the skull volume to the "complete_skull" folder in the output_path
        skull_path = os.path.join(output_path, 'complete_skulls', f'{patient_name}_skull.nii.gz')
        shutil.copy(skull_volume_path, skull_path)

        for i in range(num_defected_skulls):
            skull_volume = nib.load(skull_volume_path).get_fdata()
            skull_with_defects, implant, defect_characteristics = generate_skull_defects(skull_volume, max_sphere_radius=50, max_rectangle_size=70)
            affine = nib.load(skull_volume_path).affine
            output_defect_skull_path = os.path.join(output_path, 'defected_skulls', f'{patient_name}_defects_{i}.nii.gz')
            nib.save(nib.Nifti1Image(skull_with_defects, affine), output_defect_skull_path)
            print(f"Skull with defects saved to {output_path}")
            characteristics_csv_path = os.path.join(output_path, 'defects_characteristics', f"{patient_name}_defects_characteristics{i}.csv")
            pd.DataFrame(defect_characteristics).to_csv(characteristics_csv_path, index=False)
            print(f"Defect characteristics saved to {characteristics_csv_path}")
            output_implant_path = os.path.join(output_path, 'implants', f'{patient_name}_implant_{i}.nii.gz')
            nib.save(nib.Nifti1Image(implant, affine), output_implant_path)
            print(f"Implant volume saved to {output_implant_path}")



if __name__ == "__main__":
    # Example usage:
    # skull_volume_path = 'data/skull.nii.gz'
    # skull_volume = nib.load(skull_volume_path).get_fdata()
    #
    # for i in range(5):
    #     skull_with_defects, implant, defect_characteristics = generate_skull_defects(skull_volume, max_sphere_radius=50, max_rectangle_size=50)
    #     affine = nib.load(skull_volume_path).affine
    #     output_path = f'data/skull_subtracted_{i}.nii.gz'
    #     nib.save(nib.Nifti1Image(skull_with_defects, affine), output_path)
    #     print(f"Skull with defects saved to {output_path}")
    #     characteristics_csv_path = f'data/defect_characteristics_{i}.csv'
    #     pd.DataFrame(defect_characteristics).to_csv(characteristics_csv_path, index=False)
    #     print(f"Defect characteristics saved to {characteristics_csv_path}")
    #     implant_path = f'data/implant_{i}.nii.gz'
    #     nib.save(nib.Nifti1Image(implant, affine), implant_path)
    #     print(f"Implant volume saved to {implant_path}")

    dataset_path = "data/cq500_resampled_filtered/"
    output_path = "data/cq500_20_defects/"
    num_defect_skulls = 20

    create_defects_on_cq500_skulls(dataset_path, num_defect_skulls, output_path)