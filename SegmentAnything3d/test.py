import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

pcd_path = "/home/t/Desktop/work/datasets/results2/segmented_point_cloud6.ply"  # Update this path to your actual PCD file path
pcd = o3d.io.read_point_cloud(pcd_path)

o3d.visualization.draw_geometries([pcd])