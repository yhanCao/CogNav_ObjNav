import os, sys
import torch
import numpy as np
import omegaconf
from omegaconf import DictConfig
from pathlib import Path
import yaml
import open3d as o3d
import cv2
import pickle
import copy

radian = 180 / np.pi

def get_new_pose_from_rel_pose_batch(pose, rel_pose_change):
    pose[:, 1] += rel_pose_change[:, 0] * torch.sin(pose[:, 2] / radian) +\
                  rel_pose_change[:, 1] * torch.cos(pose[:, 2] / radian)
    pose[:, 0] += rel_pose_change[:, 0] * torch.cos(pose[:, 2] / radian) -\
                  rel_pose_change[:, 1] * torch.sin(pose[:, 2] / radian)

    pose[:, 2] += rel_pose_change[:, 2] * radian
    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

    return pose

def _object_query_constructor(objects):
    """
    Construct a query string based on a list of objects

    Args:
        objects: torch.tensor of object indices contained in a room

    Returns:
        str query describing the room, eg "This is a room containing
            toilets and sinks."
    """
    assert len(objects) > 0
    query_str = "This room contains "
    names = []
    for ob in objects:
        names.append(ob)
    if len(names) == 1:
        query_str += names[0]
    elif len(names) == 2:
        query_str += names[0] + " and " + names[1]
    else:
        for name in names[:-1]:
            query_str += name + ", "
        query_str += "and " + names[-1]
    query_str += "."
    return query_str


def process_cfg(cfg: DictConfig):
    cfg.dataset_root = Path(cfg.dataset_root)
    cfg.dataset_config = Path(cfg.dataset_config)

    if cfg.dataset_config.name != "multiscan.yaml":
        print(f"Setting image height and width to {cfg.image_height} x {cfg.image_width}")
    else:
        assert cfg.image_height is not None and cfg.image_width is not None, \
            "For multiscan dataset, image height and width must be specified"

    return cfg

def load_config(path, default_path=None):
    """
    Loads config file.
    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.
    Returns:
        cfg (dict): config dict.
    """
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    inherit_from = cfg_special.get('inherit_from')

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    """
    Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated.
        dict2 (dict): second dictionary which entries should be used.
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def camera_param_to_camera_matrix(camera_param):
    f = camera_param.f
    cx = camera_param.xc
    cy = camera_param.zc
    camera_matrix = np.array([[f, 0.0, cx, 0.0],
                              [0.0, f, cy, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]])
    return camera_matrix

def bev_pose_to_4x4_matrix(points_pose):
    x, y, theta = points_pose
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    T = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R_homogeneous = np.eye(4)
    R_homogeneous[:3, :3] = R

    points_pose_matrix = np.dot(T, R_homogeneous)
    return points_pose_matrix


def debug_show_pcd_array(pcd_array):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='debug_show_pcd')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array.reshape(-1, 3))
    pcd.estimate_normals()
    vis.add_geometry(pcd)
    vis.run()

def debug_write_pcd_array(pcd_array, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array.reshape(-1, 3))
    pcd.estimate_normals()
    o3d.io.write_point_cloud(path, pcd)

def debug_show_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='debug_show_pcd')
    vis.add_geometry(pcd)
    vis.run()

def debug_write_pcd(pcd, path):
    o3d.io.write_point_cloud(path, pcd)

def preprocess_depth(depth, min_d, max_d):
    depth = depth[:, :, 0] * 1

    mask2 = depth > 0.99
    mask1 = depth == 0
    depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0
    depth[mask1] = 0.
    depth[mask2] = 0.

    return depth

def save_ori_obs_rgbd(ori_obs, idx, path='outputs'):
    path_dir_save = os.path.join(path, 'save_ori_obs_rgbd')
    if not os.path.exists(path_dir_save):
        os.makedirs(path_dir_save)

    save_obs = ori_obs.transpose(1, 2, 0)
    rgb = save_obs[:, :, :3].astype(np.uint8)
    save_obs[:, :, 3] = preprocess_depth(save_obs[:, :, 3, None], 0.5, 5.0)
    _rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # 保存npy
    np.save(os.path.join(path_dir_save, '{}.npy'.format(idx)), save_obs[:, :, :4])

    cv2.imwrite(os.path.join(path_dir_save, '{}.jpg'.format(idx)), _rgb)

    print('save_ori_obs_rgbd')

def save_intermediate_element_for_SGG(obs, pcd_frame, idx,pose,cam_K, path='outputs', flag_save_rgb=True):
    image_dir_save = os.path.join(path, 'image')
    if not os.path.exists(image_dir_save):
        os.makedirs(image_dir_save)
    pkl_dir_save = os.path.join(path, 'pkl')
    if not os.path.exists(pkl_dir_save):
        os.makedirs(pkl_dir_save)
    save_dict = {}

    _obs = obs.squeeze().permute(1, 2, 0)
    depth = preprocess_depth(_obs[:, :, 3, None], 0.5, 5.0)
    rgb = _obs[:, :, :3]


    _pcd_frame = {
        'points': pcd_frame.cpu().numpy()
    }
    save_dict['depth'] = depth
    save_dict['rgb'] = rgb
    save_dict['pose'] = pose
    save_dict['cam_K'] = cam_K
    save_dict['pcd_frame'] = _pcd_frame
    save_dict['idx'] = idx
    with open(os.path.join(pkl_dir_save, 'save_dict_{}.pkl'.format(idx)), 'wb') as tf:
        pickle.dump(save_dict, tf)
    if flag_save_rgb:
        save_rgb = rgb.cpu().detach().numpy().astype(np.uint8)
        save_rgb = cv2.cvtColor(save_rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_dir_save, '{}.jpg'.format(idx)), save_rgb)





