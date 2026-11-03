import os
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Define the folder containing the .csv files
csv_folder = "data/defects_characteristics"  # Change this path

# Initialize lists to store combined data
all_volumes = []
defect_counts = []
centers = []

# Loop through all CSV files in the folder
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder, file)
        df = pd.read_csv(file_path, delimiter=",")  # Adjust delimiter if needed

        # Convert defects column from string to dictionary/list
        df["defects"] = df["defects"].apply(ast.literal_eval)
        # Track number of defects per skull
        defect_counts.append(df["num_defects"][0])

        for _, row in df.iterrows():
            defects = row["defects"]

            # Ensure the defect data is in list format
            if isinstance(defects, dict):
                defects = [defects]

            for defect in defects:
                all_volumes.append(defect["volume"])
                centers.append((defect["center_x"], defect["center_y"], defect["center_z"]))

# Convert centers list to NumPy array for easier handling
centers = np.array(centers)

# --- Boxplot for defect volume distribution ---
plt.figure(figsize=(8, 6))
sns.boxplot(y=all_volumes)
plt.title("Defect Volume Distribution")
plt.ylabel("Volume (voxels)")
plt.show()

# --- Bar plot for number of defective skulls with 1, 2, or 3 defects ---
plt.figure(figsize=(8, 6))
sns.countplot(x=defect_counts)
plt.title("Number of Defective Skulls by Defect Count")
plt.xlabel("Number of Defects")
plt.ylabel("Skull Count")
plt.xticks([1, 2, 3])  # Ensure correct labels
plt.show()

# --- 3D Scatter plot for defect centers ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r', marker='o')

ax.set_xlim([0, 256])
ax.set_ylim([0, 256])
ax.set_zlim([0, 256])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Point Cloud of Defect Centers")

plt.show()
