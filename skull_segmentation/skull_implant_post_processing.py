import os
import nrrd
import nibabel as nib
import numpy as np
import open3d as o3d
import diplib as dp
import mcubes
import pymesh


def voxel_to_point_cloud(voxel):
    # Get the indices of non-zero elements in the voxel representation
    indices = np.nonzero(voxel)

    # Create a point cloud array
    point_cloud = np.column_stack(indices)

    return point_cloud


def point_cloud_to_voxel(point_cloud, shape):
    voxel = np.zeros(shape, dtype=bool)
    indices = tuple(point_cloud.T.astype(int))
    voxel[indices] = 1
    return voxel


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.4, 0.4, 0.4])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def clear_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    return mesh


def check_normal_direction(normal_vectors, centerpoint_defect, centerpoint_implant):
    # Convert the normal vectors and point to numpy arrays
    normal_vectors = np.array(normal_vectors)
    centerpoint_defect = np.array(centerpoint_defect)
    centerpoint_implant = np.array(centerpoint_implant)

    # Calculate the vector from the origin to the point
    vector_to_point = centerpoint_defect - centerpoint_implant

    # Calculate the dot products between the normal vectors and the vector to the point
    dot_products = np.dot(normal_vectors, vector_to_point)

    return dot_products > 0


def points_within_distance_from_surface(surface_mesh, pcd, distance_threshold):
    # Create a raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(surface_mesh)
    _ = scene.add_triangles(mesh)

    # Define query points
    query_points = np.asarray(pcd).astype(np.float32)

    # Compute the signed distance from each point to the surface
    distances = scene.compute_distance(query_points)
    distances = distances.numpy()

    # Filter the points based on the distance threshold
    idx = distances < distance_threshold
    within_distance_points = [point for distance, point in zip(distances, pcd) if distance < distance_threshold]
    within_distance_points = np.asarray(within_distance_points)
    return within_distance_points, idx


def convert_nii_to_nrrd(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace('.nii.gz', '.nrrd'))
            try:
                # Load the NIfTI file
                nii_img = nib.load(input_path)
                nii_data = nii_img.get_fdata()

                # Save as .nrrd
                nrrd.write(output_path, nii_data)
                print(f"Converted: {input_path} -> {output_path}")
            except Exception as e:
                print(f"Converting error: {input_path}: {e}")

def process_implants(input_folder, defective_path, output_folder, viz=False):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith('.nrrd') or file.endswith('.nii.gz'):
            implant_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace('.nrrd', '_final.stl').replace('.nii.gz', '_final.stl'))
            try:
                print(f"[---------- Processing {file} ----------]")
                print("[1] Reading implant ...")
                if file.endswith('.nii.gz'):
                    nii_img = nib.load(implant_path)
                    implant, _ = nii_img.get_fdata(), None
                else:
                    implant, _ = nrrd.read(implant_path)

                print("[2] Binarizing ...")
                implant = np.asarray(dp.Closing(implant, se=dp.SE((8, 8, 8), "rectangular")))
                implant = np.asarray(dp.Closing(implant, se=dp.SE((5, 5, 5), "rectangular")))

                print("[3] Removing outliers ...")
                implant_pc = o3d.geometry.PointCloud()
                implant_pc.points = o3d.utility.Vector3dVector(np.column_stack(np.nonzero(implant)))
                cleared, ind = implant_pc.remove_radius_outlier(nb_points=1500, radius=10)
                if viz:
                    inlier_cloud = cleared.select_by_index(ind)
                    outlier_cloud = cleared.select_by_index(ind, invert=True)
                    outlier_cloud.paint_uniform_color([1, 0, 0])
                    inlier_cloud.paint_uniform_color([0.4, 0.4, 0.4])
                    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

                print("[4] Extracting surface mesh ...")
                implant = np.zeros(implant.shape, dtype=bool)
                indices = tuple(np.asarray(cleared.points).T.astype(int))
                implant[indices] = 1
                vertices, triangles = mcubes.marching_cubes(implant, 0)
                mcubes.export_obj(vertices, triangles, output_path.replace('_final.stl', '.obj'))

                print("[5] Smoothing the mesh ...")
                implant_mesh = o3d.io.read_triangle_mesh(output_path.replace('_final.stl', '.obj'))
                implant_mesh = implant_mesh.filter_smooth_simple(number_of_iterations=5)
                implant_mesh = implant_mesh.remove_degenerate_triangles()
                implant_mesh.compute_vertex_normals()

                print("[6] Saving final mesh ...")
                o3d.io.write_triangle_mesh(output_path, implant_mesh)
                print(f"Final mesh saved in: {output_path}")

            except Exception as e:
                print(f"Error in processing {implant_path}: {e}")

# Input and output paths
input_folder = 'data/clinical_cases/nifti_implant_centered'
output_folder_nrrd = 'data/clinical_cases/nrrd_implant_centered'
output_folder_processed = 'data/clinical_cases/nifti_implant_centered_processed'
defective_path = "data/clinical_cases/stl_thinned_implant"

# Convert .nii.gz files to .nrrd
convert_nii_to_nrrd(input_folder, output_folder_nrrd)

# Process implants
process_implants(output_folder_nrrd, defective_path, output_folder_processed)