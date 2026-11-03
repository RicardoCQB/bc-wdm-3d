'''This script will read the dataset info csv into a pandas dataframe and use the column "use" to filter the rows of patients
that will be usable for the dataset. The rows of patients that possess the value "True" in the column "use" will be used
and copied to the output_path. their original path is the dataset_path and the filename in the column "file". And
 they will copied to the output_path with the same filename. The dataset_path is the path to the dataset folder, the
 dataset_csv_path is the path to the csv file and the output_path is the path to the output folder. The csv file must
 have the columns "file" and "use". The copied files will be copied to the output_path but put inside a folder within
 with the name of file.'''

import pandas as pd
import os
import csv
import shutil

from scipy.cluster.hierarchy import complete


def filter_dataset(database_path, database_path_csv, output_path):
    df = pd.read_csv(database_path_csv)
    for index, row in df.iterrows():
        if row['use'] == True:
            file_path = os.path.join(database_path, row['file'])
            file_name = os.path.basename(file_path)
            folder_name = os.path.splitext(file_name)[0]
            folder_path = os.path.join(output_path, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            shutil.copy(file_path, folder_path)


def create_csv_dataset_filepaths(dataset_path, csv_output_path):
    '''
    For each patient in the dataset_path folder "complete_skulls", it creates a line with the first column
    being the path to the patient's skull and the second column being the path to the patient's skull with
    defects. Each patient will have some skulls with defects located in the "defected_skulls" folder.
    The column with the file path of the patient's complete skull will be repeated for each
    defect's path on the second column. The csv file will be saved in the csv_output_path.
    '''

    # Get the list of patients
    complete_skulls_path = os.path.join(dataset_path, 'complete_skulls')
    patients_complete_skull_file_names = os.listdir(complete_skulls_path)
    patients_names = [patient.split('_')[0] for patient in patients_complete_skull_file_names]
    defected_skulls_path = os.path.join(dataset_path, 'defected_skulls')

    # Create .csv with header "image_path missing mode"
    with open(csv_output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path', 'missing_path', 'mode'])

    for i, patient_name in enumerate(patients_names):
        patient_defected_skulls_paths = []
        for defected_skull_file_name in os.listdir(defected_skulls_path):
            if patient_name in defected_skull_file_name:
                patient_defected_skulls_paths.append(os.path.join(defected_skulls_path, defected_skull_file_name))

        # Write rows on the csv file, each row will have the complete skull path and the defected skull path
        with open(csv_output_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for defected_skull_path in patient_defected_skulls_paths:
                csv_writer.writerow([os.path.join(complete_skulls_path, patients_complete_skull_file_names[i]),
                                     defected_skull_path, 'train'])


if __name__ == "__main__":
    # dataset_path = 'data/dataset_resampled'
    # database_path_csv = 'data/database.csv'
    # output_path = 'data/cq500_resampled_filtered'
    # filter_dataset(dataset_path, database_path_csv, output_path)

    # This creates the csv file with the paths to the images and the condition images of the CQ500 dataset
    dataset_path = "Datasets/cq500_20_defects/"
    csv_output_path = "Datasets/cq500_20_defects/bc_wdm_skullbreak_cq20defects_train.csv"
    create_csv_dataset_filepaths(dataset_path, csv_output_path)