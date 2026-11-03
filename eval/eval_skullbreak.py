''' This script is used to evaluate the skullbreaker dataset. It will compare each "{defect}_sample.nii.gz" to the
 " {defect}_target.nii.gz image and calculate de DSC and HD95 for each patient and write it to a csv file. '''

import sys
import os
import numpy as np
import pandas as pd
import nibabel as nib
import csv
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import label
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure

sys.path.append(".")

def calculate_datataset_metrics(path, implants_path):
    header = ['patient', 'defect', 'DSC', 'HD95']
    dataset_name = path.split('/')[-2]

    # Create the csv file
    with open(f'{dataset_name}_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    for patient in os.listdir(path):
        if "CT" not in patient:
            patient_path = os.path.join(path, patient)
            if os.path.isdir(patient_path):
                print(f"Calculating metrics for patient {patient}")
                patient_metrics = calculate_patient_metrics(patient_path, implants_path, patient)
                print(patient_metrics)
                with open(f'{dataset_name}_metrics.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(patient_metrics)



def calculate_patient_metrics(patient_path, implants_path, patient_name):
    patient_metrics = []
    defect_names = ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']
    for defect in defect_names:
        sample = nib.load(os.path.join(patient_path, f'{defect}_sample_implant.nii.gz')).get_fdata()
        target = nib.load(os.path.join(implants_path, defect, f'{patient_name}.nii.gz')).get_fdata()

        dsc = dc(sample, target)
        hd95_metric = hd95(sample, target)

        patient_metrics.append([patient_name, defect, dsc, hd95_metric])

    return patient_metrics


def keep_largest_connected_component(volume):
    # Label connected components
    labeled_array, num_features = label(volume)

    # Find the largest connected component
    largest_component = 0
    largest_size = 0
    for component in range(1, num_features + 1):
        component_size = np.sum(labeled_array == component)
        if component_size > largest_size:
            largest_size = component_size
            largest_component = component

    # Create a binary mask for the largest connected component
    largest_component_mask = (labeled_array == largest_component).astype(np.float32)

    return largest_component_mask

def subtract_sample_defect(sample_path, defect_path):
    # Print paths and shapes of both images
    print(f"Sample path: {sample_path}")
    print(f"Defect path: {defect_path}")

    sample = nib.load(sample_path).get_fdata()
    defect = nib.load(defect_path).get_fdata()

    # Print shapes of both images
    print(f"Sample shape: {sample.shape}")
    print(f"Defect shape: {defect.shape}")
    print(f'Sample min and max: {sample.min()} - {sample.max()}')
    print(f'Defect min and max: {defect.min()} - {defect.max()}')

    sample_implant = sample - defect

    #sample_implant = keep_largest_connected_component(sample_implant)

    return sample_implant

def subtract_patient_defects(patient_path_samples, defects_path, patient_name):
    defect_names = ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']
    for defect in defect_names:
        sample_path = os.path.join(patient_path_samples, f'{defect}_sample.nii.gz')
        defect_path = os.path.join(defects_path, defect, f'{patient_name}.nii.gz')
        sample_inplant = subtract_sample_defect(sample_path, defect_path)
        output_path = os.path.join(patient_path_samples, f'{defect}_sample_implant.nii.gz')
        img = nib.Nifti1Image(sample_inplant, np.eye(4))
        nib.save(img=img, filename=output_path)

def subtract_dataset_defects(path, defects_path):
    for patient in os.listdir(path):
        # Ignore patient folder if it contains "CT"
        if "CT" not in patient:
            patient_path = os.path.join(path, patient)
            if os.path.isdir(patient_path):
                print(f"Subtracting defects for patient {patient}")
                subtract_patient_defects(patient_path, defects_path, patient)


def calculate_dsc(volume1, volume2):
    # Ensure the volumes are binary
    volume1 = (volume1 > 0.5).astype(np.float32)
    volume2 = (volume2 > 0.5).astype(np.float32)

    # Calculate the DSC
    intersection = np.sum(volume1 * volume2)
    volume1_sum = np.sum(volume1)
    volume2_sum = np.sum(volume2)
    dsc = (2. * intersection) / (volume1_sum + volume2_sum)

    return dsc

def calculate_hd95(volume1, volume2):
    # Ensure the volumes are binary
    volume1 = (volume1 > 0.5).astype(np.float32)
    volume2 = (volume2 > 0.5).astype(np.float32)

    # Get the coordinates of the non-zero elements
    coords1 = np.argwhere(volume1)
    coords2 = np.argwhere(volume2)

    # Calculate the directed Hausdorff distances
    hd1 = directed_hausdorff(coords1, coords2)[0]
    hd2 = directed_hausdorff(coords2, coords1)[0]

    # Calculate the 95th percentile of the Hausdorff distance
    hd95 = np.percentile([hd1, hd2], 95)

    return hd95

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = result > 0.5
    reference = reference > 0.5
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds

def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`assd`
    :func:`asd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)

    return hd95

def dc(input1, input2):
    """
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    input1 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```input1``` and the
        object(s) in ```input2```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric.
    """
    print(f'Input1 shape: {input1.shape}')
    print(f'Input2 shape: {input2.shape}')
    input1 = input1 > 0.5
    input2 = input2 > 0.5

    intersection = np.count_nonzero(input1 & input2)

    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def bdc(implant_1, implant_2, defective_skull, voxelspacing=None, distance=10):
    """
    Border Dice coefficient
    Measures how well a predicted implant fits around border of defective skull by computing the Dice coefficient
    between predicted and ground-truth implant only for voxels close to the defective skull.
    Parameters
    ----------
    implant_1 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    implant_2 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    defective_skull : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    distance : integer
        Distance in milimeters at which voxels are considered to be "border".
    Returns
    -------
    dc : float
        The Dice coefficient between the border of ```implant_1``` and the border
        of ```implant_2```. It ranges from 0 (no overlap) to 1 (perfect overlap).
    """

    dt = distance_transform_edt(~(defective_skull > 0), sampling=voxelspacing)

    implant_1_masked = implant_1.copy()
    implant_2_masked = implant_2.copy()

    implant_1_masked[dt > distance] = 0
    implant_2_masked[dt > distance] = 0

    return dc(implant_1_masked, implant_2_masked)



if __name__ == '__main__':

    # Define the path to the skullbreaker dataset
    # path = 'results/skullbreaker_unet_50000_mse/'
    # implants_path = 'data/Skullbreak_256_nii_gz/implant/'
    # defects_path = 'data/Skullbreak_256_nii_gz/defective_skull/'

    #path = 'results/skullbreaker_unet_250k_mse_10defect/'
    path2 = 'results/skullbreaker_unet_150k_mse_5defect/'
    implants_path = 'data/Skullbreak_256_nii_gz/implant/'
    defects_path = 'data/Skullbreak_256_nii_gz/defective_skull/'

    # Subtract the defects from the samples
    # subtract_dataset_defects(path, defects_path)
    # calculate_datataset_metrics(path, implants_path)

    subtract_dataset_defects(path2, defects_path)
    calculate_datataset_metrics(path2, implants_path)