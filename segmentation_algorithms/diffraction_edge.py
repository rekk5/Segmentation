from pyntcloud import PyntCloud 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys
import pdb
import open3d as o3d

# Load point cloud
pcd1 = PyntCloud.from_file("/home/t/Desktop/work/appartment_open3d/appartment_cloud.pcd")
output_dir = "/home/t/Desktop/work/appartment_open3d/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Define hyperparameters
k_n = 50
thresh = 0.03

# Initialize numpy array for point cloud data
pcd_np = np.zeros((len(pcd1.points), 6))

# Find neighbors and calculate eigenvalues
kdtree_id = pcd1.add_structure("kdtree")
k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id) 
ev = pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

# Extract x, y, z and eigenvalues
x = pcd1.points['x'].values 
y = pcd1.points['y'].values 
z = pcd1.points['z'].values 
e1 = pcd1.points['e3('+str(k_n+1)+')'].values
e2 = pcd1.points['e2('+str(k_n+1)+')'].values
e3 = pcd1.points['e1('+str(k_n+1)+')'].values

# Calculate sigma values and segment the point cloud
sum_eg = np.add(np.add(e1, e2), e3)
sigma_value = np.divide(e1, sum_eg)
sigma = sigma_value > thresh

# Visualize segmented point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x, y, z, c=sigma, cmap='jet')
fig.colorbar(img) 
plt.show() 

# Prepare the data for PyntCloud and save the edges
thresh_min = sigma_value < thresh
sigma_value[thresh_min] = 0
thresh_max = sigma_value > thresh
sigma_value[thresh_max] = 255

pcd_np[:, 0] = x
pcd_np[:, 1] = y
pcd_np[:, 2] = z
pcd_np[:, 3] = sigma_value

edge_np = np.delete(pcd_np, np.where(pcd_np[:, 3] == 0), axis=0) 

clmns = ['x', 'y', 'z', 'red', 'green', 'blue']
pcd_pd = pd.DataFrame(data=pcd_np, columns=clmns)
pcd_pd['red'] = sigma_value.astype(np.uint8)

# Create PyntCloud objects
pcd_points = PyntCloud(pcd_pd)
edge_points = PyntCloud(pd.DataFrame(data=edge_np, columns=clmns))

# Save point cloud with painted edges and only edge points
pcd_points.to_file(output_dir + 'pointcloud_edges1.ply')  
edge_points.to_file(output_dir + 'edges1.ply')
