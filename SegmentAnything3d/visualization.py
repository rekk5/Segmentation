import torch
from util import visualize_partition

# Load your generated segmented point cloud
segmented_point_cloud = torch.load('/home/t/Desktop/work/datasets/results2/segmented_point_cloud.pth')

# Extract coordinates and group IDs from the loaded point cloud data
coord = segmented_point_cloud['coord']
group_id = segmented_point_cloud['group']

# Specify the path to save the visualization
save_path = '/home/t/Desktop/work/datasets/results2/segmented_point_cloud6.ply'

# Visualize and save the segmented point cloud
visualize_partition(coord, group_id, save_path)
