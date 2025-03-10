import open3d as o3d
import numpy as np

if __name__ == "__main__":
    points = np.load("model_outputs/model_output_v01.npy")
    points = points.squeeze()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
