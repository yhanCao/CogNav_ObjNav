# @brief:  通过保存出来的点云，对场景进行点云重建，然后将其中识别出来的物体通过不同颜色的方式标出来。
import os, sys
import open3d as o3d

import os, sys
import open3d as o3d
import numpy as np
import pickle as pkl
from slam.utils import extractVoxelCenterfromVoxelmap,selectPointsFromVoxel
from detectron2.data import MetadataCatalog
import matplotlib.colors as mcolors
import argparse
# metadata = MetadataCatalog.get('coco_2017_train_panoptic')
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

def read_pkl(path_file):
    output = open(path_file, 'rb')
    res = pkl.load(output)

    for obj in res['objects']:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(obj['pcd']['points'])
        point_cloud.colors = o3d.utility.Vector3dVector(obj['pcd']['colors'])
        obj['pcd'] = point_cloud

        if 'min_bound' in obj['bbox']:
            obj['bbox'] = o3d.geometry.AxisAlignedBoundingBox(
                min_bound = obj['bbox']['min_bound'],
                max_bound = obj['bbox']['max_bound']
            )
        elif 'center' in obj['bbox']:
            obj['bbox'] = o3d.geometry.OrientedBoundingBox(
                center = obj['bbox']['center'],
                R = obj['bbox']['R'],
                extent = obj['bbox']['extent']
            )

    for obj in res['bg_objects']:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(res['bg_objects'][obj]['pcd']['points'])
        point_cloud.colors = o3d.utility.Vector3dVector(res['bg_objects'][obj]['pcd']['colors'])
        res['bg_objects'][obj]['pcd'] = point_cloud

        if 'min_bound' in res['bg_objects'][obj]['bbox']:
            res['bg_objects'][obj]['bbox'] = o3d.geometry.AxisAlignedBoundingBox(
                min_bound = res['bg_objects'][obj]['bbox']['min_bound'],
                max_bound = res['bg_objects'][obj]['bbox']['max_bound']
            )
        elif 'center' in res['bg_objects'][obj]['bbox']:
            res['bg_objects'][obj]['bbox'] = o3d.geometry.OrientedBoundingBox(
                center = res['bg_objects'][obj]['bbox']['center'],
                R = res['bg_objects'][obj]['bbox']['R'],
                extent = res['bg_objects'][obj]['bbox']['extent']
            )

    return res

def draw_voxel_objects(objects,bg_objects,idx,pcd_path):
    pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="test_poses_with_pcd")
    points=np.empty([0,3])
    normals=np.empty([0,3])
    colors=np.empty([0,3])
    for i, obj in enumerate(objects):
        coordinates = extractVoxelCenterfromVoxelmap(obj['voxel_index'])
        points = np.concatenate((points,coordinates),axis=0)
        normals = np.concatenate((normals,np.ones_like(coordinates)),axis=0)
        colors = np.concatenate((colors,np.ones_like(coordinates)*color_proposals[i%len(color_proposals)]),axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(pcd_path+str(idx)+".ply", pcd)

def draw_pcd_objects(objects,bg_objects,idx,pcd_path):
    pcd = o3d.geometry.PointCloud()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(window_name="test_poses_with_pcd")
    points=np.empty([0,3])
    normals=np.empty([0,3])
    colors=np.empty([0,3])
    for i, obj in enumerate(objects):
        coordinates = selectPointsFromVoxel(obj)
        points = np.concatenate((points,coordinates),axis=0)
        normals = np.concatenate((normals,np.ones_like(coordinates)),axis=0)
        colors = np.concatenate((colors,np.ones_like(coordinates)*color_proposals[i%len(color_proposals)]),axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(pcd_path+"semantic.ply", pcd)
def draw_pcd_with_objs(pose, pcd, img, objects):
    ln = len(pose)
    ans_pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="test_poses_with_pcd")
    for i, obj in enumerate(objects):
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = obj['pcd'].points
        tmp_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array(color_proposals[i]), (np.array(obj['pcd'].points).shape[0], 1)))
        # print('3333333')
        ans_pcd += tmp_pcd
    for i in range(ln):
        print('i:   {}'.format(i))
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(pcd[i]['points'].reshape(-1, 3))
        tmp_pcd.colors = o3d.utility.Vector3dVector(img[i].cpu().numpy().reshape(-1, 3) / 255.)
        # tmp_pcd.transform(pose[i].cpu().numpy())
        ans_pcd += tmp_pcd

    vis.add_geometry(ans_pcd)
    vis.run()
    vis.destroy_window()
    print('draw_pcd')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='dataset path')
    args = parser.parse_args()
    begin, end = -1, 34
    pose, pcd, img = [], [], []
    for i in range(begin, end+1, 1):
        pkl_dict = read_pkl(os.path.join(args.path, 'save_dict_{}.pkl'.format(i)))
        pose.append(pkl_dict['pose'])
        pcd.append(pkl_dict['pcd_frame'])
        img.append(pkl_dict['rgb'])
        
    draw_pcd_with_objs(pose, pcd, img, pkl_dict['objects'])