"""
Main Script

Author: Yunhan Yang (yhyang.myron@gmail.com)
"""

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
import multiprocessing as mp
import pointops
import random
import argparse
import yaml

from scipy.spatial.transform import Rotation as R
from segment_anything import build_sam, SamAutomaticMaskGenerator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image
from os.path import join
from util import *


def pcd_ensemble(org_path, new_path, data_path, vis_path):
    new_pcd = torch.load(new_path)
    new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
    with open(org_path) as f:
        segments = json.load(f)
        org_pcd = np.array(segments['segIndices'])
    match_inds = [(i, i) for i in range(len(new_pcd))]
    new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
    print(new_group.shape)
    data = torch.load(data_path)
    visualize_partition(data["coord"], new_group, vis_path)


def get_sam(image, mask_generator):
    masks = mask_generator.generate(image)
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        # print(masks[i]["predicted_iou"])
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1
    return group_ids


def get_pcd(rgb_path, color_name, depth_name, pose_path, mask_generator, save_2dmask_path):
    intrinsic_path = join(rgb_path, 'calibration', 'calib_color.yaml')
    depth_intrinsic, depth_to_color = load_intrinsic_from_yaml(intrinsic_path)

    timestamp = color_name.split('.')[0]

    depth = join(rgb_path, 'depth', depth_name)
    color = join(rgb_path, 'color', color_name)

    # Load depth and color images
    depth_img = cv2.imread(depth, -1)
    if depth_img is None:
        print(f"Failed to load depth image: {depth}")
        return None
    
    color_image = cv2.imread(color)
    if color_image is None:
        print(f"Failed to load color image: {color}")
        return None

    mask = (depth_img != 0)

    if mask_generator is not None:
        group_ids = get_sam(color_image, mask_generator)
        if not os.path.exists(save_2dmask_path):
            os.makedirs(save_2dmask_path)
        img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
        img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))
    else:
        group_path = join(save_2dmask_path, color_name[0:-4] + '.png')
        img = Image.open(group_path)
        group_ids = np.array(img, dtype=np.int16)

    # Process point cloud
    color_image = np.reshape(color_image[mask], [-1, 3])
    group_ids = group_ids[mask]
    colors = np.zeros_like(color_image)
    colors[:, 0] = color_image[:, 2]
    colors[:, 1] = color_image[:, 1]
    colors[:, 2] = color_image[:, 0]

    depth_shift = 1000.0
    x, y = np.meshgrid(np.linspace(0, depth_img.shape[1] - 1, depth_img.shape[1]), 
                       np.linspace(0, depth_img.shape[0] - 1, depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:, :, 0] = x
    uv_depth[:, :, 1] = y
    uv_depth[:, :, 2] = depth_img / depth_shift
    uv_depth = np.reshape(uv_depth, [-1, 3])
    uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()

    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]

    points = np.ones((uv_depth.shape[0], 4))
    points[:, 0] = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx
    points[:, 1] = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy
    points[:, 2] = uv_depth[:, 2]

    # Transform points from depth camera space to color camera space
    points_color_space = np.dot(points, np.transpose(depth_to_color))

    # Load and apply the pose
    pose = load_pose_from_txt(pose_path, timestamp)
    points_world = np.dot(points_color_space, np.transpose(pose))

    group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=points_world[:, :3], color=colors, group=group_ids)
    return save_dict



def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    group_1[group_1 != -1] += group_0.max() + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1


def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50):
    if len(index) == 1:
        return pcd_list[index[0]]

    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]

    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)

    if pcd0 is None:
        return input_dict_1 if pcd1 is not None else None
    if pcd1 is None:
        return input_dict_0

    # Create KDTree for pcd0
    pcd0_tree = o3d.geometry.KDTreeFlann(pcd0)
    
    # Get matching indices using the point cloud object
    match_inds = get_matching_indices(pcd1, pcd0_tree, 5 * voxel_size, 1)
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds)

    # Create KDTree for pcd1
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)

    # Get matching indices using the point cloud object
    match_inds = get_matching_indices(pcd0, pcd1_tree, 5 * voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds)

    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict

def get_matching_indices(source, pcd_tree, search_voxel_size, K=50):
    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds



def seg_pcd(rgb_path, save_path, mask_generator, voxel_size, voxelize, th, save_2dmask_path):
    color_dir = join(rgb_path, 'color')
    depth_dir = join(rgb_path, 'depth')
    pose_path = join(rgb_path, 'poses', 'poses_color.txt')
    
    color_names = sorted([f for f in os.listdir(color_dir) if f.endswith('.jpg')], key=lambda x: int(x.split('.')[0]))
    
    pcd_list = []
    for color_name in color_names:
        depth_name = color_name.replace('.jpg', '.png')
        pcd_dict = get_pcd(rgb_path, color_name, depth_name, pose_path, mask_generator, save_2dmask_path)
        if pcd_dict is None or len(pcd_dict["coord"]) == 0:
            continue
        
        pcd_dict = voxelize(pcd_dict)
        pcd_list.append(pcd_dict)
    
    if not pcd_list:
        print("No valid point clouds generated.")
        return
    
    # Combine point clouds if there are multiple frames
    while len(pcd_list) != 1:
        new_pcd_list = []
        for indice in pairwise_indices(len(pcd_list)):
            pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize)
            if pcd_frame is not None:
                new_pcd_list.append(pcd_frame)
        pcd_list = new_pcd_list
    
    seg_dict = pcd_list[0]
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

    scene_path = join(save_path, "segmented_point_cloud.pth")
    torch.save(seg_dict, scene_path)
    print(f"Finished processing and saved to {scene_path}")


def load_intrinsic_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        calib_data = yaml.safe_load(file)
    intrinsics = np.array(calib_data['camera_matrix']['data']).reshape(3, 3)
    depth_to_color = np.array(calib_data['depth_to_color']['data']).reshape(4, 4)
    return intrinsics, depth_to_color

def load_pose_from_txt(pose_path, timestamp):
    closest_pose = None
    closest_time_diff = float('inf')

    with open(pose_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            pose_time = float(parts[0])
            time_diff = abs(pose_time - float(timestamp))

            if time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_pose = parts[1:]

    if closest_pose is None or closest_time_diff > 0.1:
        raise ValueError(f"Timestamp {timestamp} not found or no close match in pose file.")

    translation = np.array(closest_pose[:3], dtype=np.float32)
    quaternion = np.array(closest_pose[3:], dtype=np.float32)

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


def load_pose(pose_path, timestamp):
    closest_pose = None
    closest_time_diff = float('inf')

    with open(pose_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            pose_time = float(parts[0])
            time_diff = abs(pose_time - float(timestamp))

            if time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_pose = parts[1:8]

    if closest_pose is None or closest_time_diff > 0.1:
        raise ValueError(f"Timestamp {timestamp} not found or no close match in pose file.")

    translation = np.array(closest_pose[:3], dtype=np.float32)
    quaternion = np.array(closest_pose[3:], dtype=np.float32)

    # Convert quaternion to rotation matrix
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix




def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--rgb_path', type=str, help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--sam_checkpoint_path', type=str, default='', help='the path of checkpoint for SAM')
    parser.add_argument('--scannetv2_train_path', type=str, default='scannet-preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[1280,720])
    parser.add_argument('--voxel_size', default=0.2, help='voxel size for voxelization')
    parser.add_argument('--th', default=200, help='threshold of ignoring small groups to avoid noise pixel')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    print(args)

    # Load the SAM model
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))

    # Set up voxelization
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))

    # Make sure the save path exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Call the seg_pcd function for processing
    seg_pcd(args.rgb_path, args.save_path, mask_generator, args.voxel_size, voxelize, args.th, args.save_2dmask_path)

