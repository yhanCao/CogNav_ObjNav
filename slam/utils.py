from collections import Counter
import copy
import json
import cv2

import numpy as np
from omegaconf import DictConfig
import open3d as o3d
import torch

import torch.nn.functional as F

# import faiss

from utils.general_utils import to_tensor, to_numpy, Timer
from slam.slam_classes import MapObjectList, DetectionList

from utils.ious import compute_3d_iou, compute_3d_iou_accuracte_batch, mask_subtract_contained, compute_iou_batch,compute_3d_distance,compute_3d_distancereplica
from dataset.datasets_common import from_intrinsics_matrix
def extractVoxelCenterfromVoxelmap(index,shape=(960,960,48),minbound=(0,0,-8)):
    indices_3=np.array(np.unravel_index(index, shape,order='C')).T
    coordinates = (indices_3+ minbound + 0.5) * 5 
    return coordinates
def extractVoxelIndex3(index,shape=(960,960,48),minbound=(0,0,-8)):
    indices_3=np.array(np.unravel_index(index, shape,order='C')).T
    return indices_3
def mapPcdToVoxel(points,voxel_size = 5,shape=(960,960,48),minbound=(0,0,-8)):
    ####map points to 1D voxel index
    voxel_index_3D=np.floor(points/voxel_size-minbound)
    index = voxel_index_3D >= np.array([0,0,0])
    index_real = np.where(index[:,0]&index[:,1]&index[:,2])[0]
    voxel_index_1D=voxel_index_3D[index_real,0]*shape[1]*shape[2]+voxel_index_3D[index_real,1]*shape[2]+voxel_index_3D[index_real,2]#(n)
    voxel_index_1D = voxel_index_1D.astype(np.int32)
    voxel_index = np.unique(voxel_index_1D)
    return voxel_index
def selectPointsFromVoxel(object,voxel_size = 5,shape=(960,960,48),minbound=(0,0,-8)):
    voxel_index_3D=np.floor(np.array(object['pcd'].points)/voxel_size-minbound)
    index = voxel_index_3D > np.array([0,0,0])
    index_real = np.where(index[:,0]&index[:,1]&index[:,2])[0]
    voxel_index_1D=voxel_index_3D[index_real,0]*shape[1]*shape[2]+voxel_index_3D[index_real,1]*shape[2]+voxel_index_3D[index_real,2]#(n)
    voxel_index_1D = voxel_index_1D.astype(np.int32)
    mask = np.isin(voxel_index_1D, object['voxel_index'])
    indices = index_real[np.where(mask)[0]]
    return np.array(object['pcd'].points)[indices]
def selectPcdFromVoxel(object,voxel_size = 5,shape=(960,960,48),minbound=(0,0,-8)):
    voxel_index_3D=np.floor(np.array(object['pcd'].points)/voxel_size-minbound)
    index = voxel_index_3D > np.array([0,0,0])
    index_real = np.where(index[:,0]&index[:,1]&index[:,2])[0]
    voxel_index_1D=voxel_index_3D[index_real,0]*shape[1]*shape[2]+voxel_index_3D[index_real,1]*shape[2]+voxel_index_3D[index_real,2]#(n)
    voxel_index_1D = voxel_index_1D.astype(np.int32)
    mask = np.isin(voxel_index_1D, object['voxel_index'])
    indices = index_real[np.where(mask)[0]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(object['pcd'].points)[indices])
    # pcd.normals = o3d.utility.Vector3dVector(np.array(object['pcd'].normals)[indices])
    pcd.colors = o3d.utility.Vector3dVector(np.array(object['pcd'].colors)[indices])
    return pcd
def get_classes_colors(classes):
    class_colors = {}

    # Generate a random color for each class
    for class_idx, class_name in enumerate(classes):
        # Generate random RGB values between 0 and 255
        r = np.random.randint(0, 256)/255.0
        g = np.random.randint(0, 256)/255.0
        b = np.random.randint(0, 256)/255.0

        # Assign the RGB values as a tuple to the class in the dictionary
        class_colors[class_idx] = (r, g, b)

    class_colors[-1] = (0, 0, 0)

    return class_colors

def create_or_load_colors(cfg, filename="gsa_classes_tag2text"):
    
    # get the classes, should be saved when making the dataset
    classes_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}.json"
    classes  = None
    with open(classes_fp, "r") as f:
        classes = json.load(f)
    
    # create the class colors, or load them if they exist
    class_colors  = None
    class_colors_fp = cfg['dataset_root'] / cfg['scene_id'] / f"{filename}_colors.json"
    if class_colors_fp.exists():
        with open(class_colors_fp, "r") as f:
            class_colors = json.load(f)
        print("Loaded class colors from ", class_colors_fp)
    else:
        class_colors = get_classes_colors(classes)
        class_colors = {str(k): v for k, v in class_colors.items()}
        with open(class_colors_fp, "w") as f:
            json.dump(class_colors, f)
        print("Saved class colors to ", class_colors_fp)
    return classes, class_colors
def create_object_pcd_from_pcd(pcd_array, image, mask, obj_color=None) -> o3d.geometry.PointCloud:
    pcd_array = pcd_array[mask]
    if obj_color is None:  # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else:  # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(pcd_array.shape, obj_color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
def create_object_pcd(depth_array, mask, cam_K, image, obj_color=None) -> o3d.geometry.PointCloud:
    fx, fy, cx, cy = from_intrinsics_matrix(cam_K)
    
    # Also remove points with invalid depth values
    mask = np.logical_and(mask, depth_array > 0)

    if mask.sum() == 0:
        pcd = o3d.geometry.PointCloud()
        return pcd
        
    height, width = depth_array.shape
    x = np.arange(0, width, 1.0)
    y = np.arange(0, height, 1.0)
    u, v = np.meshgrid(x, y)
    
    # Apply the mask, and unprojection is done only on the valid points
    masked_depth = depth_array[mask] # (N, )
    u = u[mask] # (N, )
    v = v[mask] # (N, )

    # Convert to 3D coordinates
    x = (u - cx) * masked_depth / fx
    y = (v - cy) * masked_depth / fy
    z = masked_depth

    # Stack x, y, z coordinates into a 3D point cloud
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)
    
    # Perturb the points a bit to avoid colinearity
    points += np.random.normal(0, 4e-3, points.shape)

    if obj_color is None: # color using RGB
        # # Apply mask to image
        colors = image[mask] / 255.0
    else: # color using group ID
        # Use the assigned obj_color for all points
        colors = np.full(points.shape, obj_color)
    
    if points.shape[0] == 0:
        import pdb; pdb.set_trace()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd
def remove_outlierone(cfg,voxel_index):
    voxel = o3d.geometry.PointCloud()
    coordinates= extractVoxelCenterfromVoxelmap(voxel_index)
    voxel.points=o3d.utility.Vector3dVector(coordinates)
    voxel_new = pcd_denoise_dbscan(voxel,cfg)
    if voxel != None :
        return mapPcdToVoxel(np.array(voxel_new.points))
    else :
        return mapPcdToVoxel(np.array(voxel.points))
def remove_outlier(cfg,objects:MapObjectList):
    deleted=[]
    for i in range(len(objects)):
        voxel = o3d.geometry.PointCloud()
        coordinates = extractVoxelCenterfromVoxelmap(objects[i]['voxel_index'])
        voxel.points=o3d.utility.Vector3dVector(coordinates)
        voxel = pcd_denoise_dbscan(voxel,cfg)
        if voxel != None :
            objects[i]['voxel_index']=mapPcdToVoxel(np.array(voxel.points))
        else :
            deleted.append(i)
    new_objects = [objects[i] for i in range(len(objects)) if i not in deleted]
    objects = MapObjectList(new_objects)
    return objects

def pcd_denoise_dbscan(pcd: o3d.geometry.PointCloud, cfg) -> o3d.geometry.PointCloud:
    pcd_clusters = pcd.cluster_dbscan(
        eps=cfg.dbscan_eps,
        min_points=cfg.dbscan_min_points,
    )
    # Convert to numpy arrays
    obj_points = np.asarray(pcd.points)

    # obj_colors = np.asarray(pcd.colors)
    pcd_clusters = np.array(pcd_clusters)

    # Count all labels in the cluster
    counter = Counter(pcd_clusters)

    # Remove the noise label
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Find the label of the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        
        # Create mask for points in the largest cluster
        largest_mask = (pcd_clusters == most_common_label)

        # Apply mask
        largest_cluster_points = obj_points[largest_mask]
        # largest_cluster_colors = obj_colors[largest_mask]
        
        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            # print("after len < 5")
            # o3d.visualization.draw_geometries([pcd])
            return None

        # Create a new PointCloud object
        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
        # largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)
        
        pcd = largest_cluster_pcd
        # print("after")
        # o3d.visualization.draw_geometries([pcd])
        return pcd
    else :
        return None
    
def calculate_pcd_overlap(x1,x2,batch_size,threshold):
    N = x1.size(0)
    M = x2.size(0)
    distances_N = torch.ones((N,), device=x1.device)*100
    distances_M = torch.ones((M,), device=x1.device)*100
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        for j in range(0, M, batch_size):
            end_j = min(j + batch_size, M)
            distances = torch.cdist(x1[i:end_i], x2[j:end_j], p=2)
            distances_N[i:end_i] = torch.min(distances_N[i:end_i],torch.min(distances,dim=1)[0])
            distances_M[j:end_j] = torch.min(distances_M[j:end_j],torch.min(distances,dim=0)[0])
            del distances
            torch.cuda.empty_cache()
    overlap1 = (distances_N< threshold).float().sum().item() / N
    overlap2 = (distances_M< threshold).float().sum().item() / M
    del distances_N,distances_M
    torch.cuda.empty_cache()
    return overlap2,overlap1
def process_pcd(pcd, cfg, run_dbscan=True):
    
    pcd = pcd.voxel_down_sample(voxel_size=cfg.downsample_voxel_size)
        
    return pcd

def get_bounding_box(cfg, pcd):
    if ("accurate" in cfg.spatial_sim_type or "overlap" in cfg.spatial_sim_type) and len(pcd.points) >= 4:
        try:
            return pcd.get_axis_aligned_bounding_box()
        except RuntimeError as e:
            print(f"Met {e}, use axis aligned bounding box instead")
            return pcd.get_axis_aligned_bounding_box()
    else:
        return pcd.get_axis_aligned_bounding_box()
def merge_obj2_into_obj1_2(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft"]:
            if isinstance(obj1[k], list) or isinstance(obj1[k], int):
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            # else:
            #     # TODO: handle other types if needed in the future
            #     raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue

    # merge pcd and bbox
    obj1['pcd'] += obj2['pcd']
    obj1['pcd'] = process_pcd(obj1['pcd'], cfg, run_dbscan=run_dbscan)
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
    
    #merge clip ft
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                       obj2['clip_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg.device)
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg.device)
    obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                       obj2['text_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    return obj1
def merge_obj2_into_obj1(cfg, obj1, obj2, run_dbscan=True):
    '''
    Merge the new object to the old object
    This operation is done in-place
    '''
    n_obj1_det = obj1['num_detections']
    n_obj2_det = obj2['num_detections']
    
    for k in obj1.keys():
        if k in ['caption']:
            # Here we need to merge two dictionaries and adjust the key of the second one
            for k2, v2 in obj2['caption'].items():
                obj1['caption'][k2 + n_obj1_det] = v2
        elif k not in ['pcd', 'bbox', 'clip_ft', "text_ft","voxel_index"]:
            if isinstance(obj1[k], list) :
                obj1[k] += obj2[k]
            elif isinstance(obj1[k], int) :
                obj1[k] += obj2[k]
            elif k == "inst_color":
                obj1[k] = obj1[k] # Keep the initial instance color
            elif k == "voxel_size":
                obj1[k] = obj1[k]
            else:
                # TODO: handle other types if needed in the future
                raise NotImplementedError
        else: # pcd, bbox, clip_ft, text_ft are handled below
            continue
    obj1['pcd'] = (obj1['pcd']+obj2['pcd']).voxel_down_sample(1)
    
    obj1['voxel_index'] = remove_outlierone(cfg,np.hstack((obj1['voxel_index'],obj2['voxel_index'])))
    obj1['bbox'] = get_bounding_box(cfg, obj1['pcd'])
    obj1['bbox'].color = [0,1,0]
        
    #merge clip ft
    obj1['clip_ft'] = (obj1['clip_ft'] * n_obj1_det +
                       obj2['clip_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['clip_ft'] = F.normalize(obj1['clip_ft'], dim=0)

    # merge text_ft
    obj2['text_ft'] = to_tensor(obj2['text_ft'], cfg.device)
    obj1['text_ft'] = to_tensor(obj1['text_ft'], cfg.device)
    obj1['text_ft'] = (obj1['text_ft'] * n_obj1_det +
                       obj2['text_ft'] * n_obj2_det) / (
                       n_obj1_det + n_obj2_det)
    obj1['text_ft'] = F.normalize(obj1['text_ft'], dim=0)
    return obj1

def compute_relationship_matrix(objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['voxel_index']
                box_j = objects[j]['voxel_index']
                common_elements = np.intersect1d(box_i, box_j)
                if len(common_elements)  == 0:
                    distance=compute_3d_distance( objects[i], objects[j])
                    if distance <1 :
                        overlap_matrix[i][j]=1
                        overlap_matrix[j][i]=1
                else :
                    overlap_matrix[i][j]=1
                    overlap_matrix[j][i]=1

    return overlap_matrix
def compute_relationship_matrixreplica(objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip diagonal elements
                box_i = objects[i]['bbox']
                box_j = objects[j]['bbox']
                iou = compute_3d_iou(box_i, box_j)
                if iou == 0:
                    distance=compute_3d_distancereplica(box_i, box_j)
                    if distance > 5 :
                        overlap_matrix[i][j]=1
                        overlap_matrix[j][i]=1
                else :
                    overlap_matrix[i][j]=1
                    overlap_matrix[j][i]=1

    return overlap_matrix
def compute_overlap_matrix(cfg, objects: MapObjectList):
    '''
    compute pairwise overlapping between objects in terms of point nearest neighbor. 
    Suppose we have a list of n point cloud, each of which is a o3d.geometry.PointCloud object. 
    Now we want to construct a matrix of size n x n, where the (i, j) entry is the ratio of points in point cloud i 
    that are within a distance threshold of any point in point cloud j. 
    '''
    n = len(objects)
    overlap_matrix = np.zeros((n, n))
    
    # Compute the pairwise overlaps
    for i in range(n-1):
        for j in range(i+1,n):
            box_i = objects[i]['bbox']
            box_j = objects[j]['bbox']
            
            # Skip if the boxes do not overlap at all (saves computation)
            iou = compute_3d_iou(box_i, box_j)
            if iou == 0:
                continue
            overlap1,overlap2 = calculate_pcd_overlap(torch.from_numpy(np.asarray(objects[i]['pcd'].points, dtype=np.float32)).cuda(),torch.tensor(np.asarray(objects[j]['pcd'].points, dtype=np.float32)).cuda(),batch_size=cfg.cal_batch_size,threshold=cfg.downsample_voxel_size)
        
            overlap_matrix[i, j] = overlap1
            overlap_matrix[j, i] = overlap2
    return overlap_matrix

def compute_overlap_matrix_2set(cfg, objects_map: MapObjectList, objects_new: DetectionList) -> np.ndarray:
    '''
    compute pairwise overlapping between two set of objects in terms of point nearest neighbor. 
    objects_map is the existing objects in the map, objects_new is the new objects to be added to the map
    Suppose len(objects_map) = m, len(objects_new) = n
    Then we want to construct a matrix of size m x n, where the (i, j) entry is the ratio of points 
    in point cloud i that are within a distance threshold of any point in point cloud j.
    '''
    m = len(objects_map)
    n = len(objects_new)
    overlap_matrix1 = torch.zeros((m, n))
    overlap_matrix2 = torch.zeros((m,n))
        
    bbox_map = objects_map.get_stacked_values_torch('bbox')
    bbox_new = objects_new.get_stacked_values_torch('bbox')
    try:
        iou = compute_3d_iou_accuracte_batch(bbox_map, bbox_new) # (m, n)
    except ValueError:
        print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
        bbox_map = []
        bbox_new = []
        for pcd in objects_map.get_values('pcd'):
            bbox_map.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        for pcd in objects_new.get_values('pcd'):
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
        bbox_map = torch.from_numpy(np.stack(bbox_map))
        bbox_new = torch.from_numpy(np.stack(bbox_new))
        
        iou = compute_iou_batch(bbox_map, bbox_new) # (m, n)
    for i in range(m):
        for j in range(n):
            if iou[i,j] < 1e-6:
                continue
            overlap1,overlap2 = calculate_pcd_overlap(torch.from_numpy(np.asarray(objects_map[i]['pcd'].points, dtype=np.float32)).cuda(),torch.tensor(np.asarray(objects_new[j]['pcd'].points, dtype=np.float32)).cuda(),batch_size=cfg.cal_batch_size,threshold=cfg.downsample_voxel_size)
            overlap_matrix1[i, j] = overlap1
            overlap_matrix2[i, j] = overlap2
    return overlap_matrix1,overlap_matrix2
def merge_overlap_objects(cfg, objects: MapObjectList, overlap_matrix: np.ndarray):
    x, y = overlap_matrix.nonzero()
    overlap_ratio = overlap_matrix[x, y]

    sort = np.argsort(overlap_ratio)[::-1]
    x = x[sort]
    y = y[sort]
    overlap_ratio = overlap_ratio[sort]

    kept_objects = np.ones(len(objects), dtype=bool)
    for i, j, ratio in zip(x, y, overlap_ratio):
        visual_sim = F.cosine_similarity(
            to_tensor(objects[i]['clip_ft']),
            to_tensor(objects[j]['clip_ft']),
            dim=0
        )
        text_sim = F.cosine_similarity(
            to_tensor(objects[i]['text_ft']),
            to_tensor(objects[j]['text_ft']),
            dim=0
        )
        if ratio > cfg.merge_overlap_thresh:
            if text_sim > cfg.merge_text_sim_thresh:
                if kept_objects[j]:
                    objects[j] = merge_obj2_into_obj1(cfg, objects[j], objects[i], run_dbscan=True)
                    kept_objects[i] = False
        else:
            break
    new_objects = [obj for obj, keep in zip(objects, kept_objects) if keep]
    objects = MapObjectList(new_objects)
    return objects

def denoise_objects(cfg, objects: MapObjectList):
    for i in range(len(objects)):
        og_object_pcd = objects[i]['pcd']
        objects[i]['pcd'] = process_pcd(objects[i]['pcd'], cfg, run_dbscan=True)
        if len(objects[i]['pcd'].points) < 4:
            objects[i]['pcd'] = og_object_pcd
            continue
        objects[i]['bbox'] = get_bounding_box(cfg, objects[i]['pcd'])
        objects[i]['bbox'].color = [0,1,0]
        
    return objects

def filter_objects(cfg, objects: MapObjectList):
    print("Before filtering:", len(objects))
    objects_to_keep = []
    for obj in objects:
        if len(obj['pcd'].points) >= cfg.obj_min_points and obj['num_detections'] >= cfg.obj_min_detections:
            objects_to_keep.append(obj)
    objects = MapObjectList(objects_to_keep)
    print("After filtering:", len(objects))
    
    return objects

def merge_objects(cfg, objects: MapObjectList):
    if cfg.merge_overlap_thresh > 0:
        overlap_matrix = compute_overlap_matrix(cfg, objects)
        print("Before merging:", len(objects))
        objects = merge_overlap_objects(cfg, objects, overlap_matrix)
        print("After merging:", len(objects))
    
    return objects

def filter_gobs(
    cfg: DictConfig,
    gobs: dict,
    image: np.ndarray,
    BG_CLASSES = ["wall", "floor", "ceiling"],
):
    # If no detection at all
    if len(gobs['xyxy']) == 0:
        return gobs
    
    # Filter out the objects based on various criteria
    idx_to_keep = []
    for mask_idx in range(len(gobs['xyxy'])):
        class_name = gobs['classes'][mask_idx]
        
        # SKip masks that are too small
        if gobs['mask'][mask_idx].sum() < max(cfg.mask_area_threshold, 10):
            continue
        if cfg.skip_bg and class_name in BG_CLASSES:
            continue

        if class_name not in BG_CLASSES:
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            bbox_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            if bbox_area > cfg.max_bbox_area_ratio * image_area:
                continue
        idx_to_keep.append(mask_idx)
    
    for k in gobs.keys():
        if isinstance(gobs[k], str) or k == "classes": # Captions
            continue
        elif isinstance(gobs[k], list):
            gobs[k] = [gobs[k][i] for i in idx_to_keep]
        elif isinstance(gobs[k], np.ndarray):
            gobs[k] = gobs[k][idx_to_keep]
        else:
            raise NotImplementedError(f"Unhandled type {type(gobs[k])}")
    
    return gobs

def resize_gobs(
    gobs,
    image
):
    n_masks = len(gobs['xyxy'])

    new_mask = []
    
    for mask_idx in range(n_masks):
        # TODO: rewrite using interpolation/resize in numpy or torch rather than cv2
        mask = gobs['mask'][mask_idx]
        if mask.shape != image.shape[:2]:
            # Rescale the xyxy coordinates to the image shape
            x1, y1, x2, y2 = gobs['xyxy'][mask_idx]
            x1 = round(x1 * image.shape[1] / mask.shape[1])
            y1 = round(y1 * image.shape[0] / mask.shape[0])
            x2 = round(x2 * image.shape[1] / mask.shape[1])
            y2 = round(y2 * image.shape[0] / mask.shape[0])
            gobs['xyxy'][mask_idx] = [x1, y1, x2, y2]
            
            # Reshape the mask to the image shape
            mask = cv2.resize(mask.astype(np.uint8), image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            new_mask.append(mask)

    if len(new_mask) > 0:
        gobs['mask'] = np.asarray(new_mask)
        
    return gobs
def gobs_to_detection_list_Replica(
    cfg, 
    image, 
    depth_array,
    cam_K, 
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    color_path = None,
    voxel_size = 5
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame. 
    '''
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES)
    
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list
    
    # Compute the containing relationship among all detections and subtract fg from bg objects
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
    n_masks = len(gobs['xyxy'])
    for mask_idx in range(n_masks):
        local_class_id = gobs['class_id'][mask_idx]
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][mask_idx]
        global_class_id = -1 if class_names is None else class_names.index(class_name)
        
        # make the pcd and color it
        camera_object_pcd = create_object_pcd(
            depth_array,
            mask,
            cam_K,
            image,
            obj_color = None
        )
        
        # It at least contains 5 points
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5): 
            continue
        
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        # global_object_pcd = process_pcd(global_object_pcd, cfg)
        voxel_index = mapPcdToVoxel(np.array(global_object_pcd.points),voxel_size=voxel_size)
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        
        if pcd_bbox.volume() < 1e-6:
            continue
        
        # Treat the detection in the same way as a 3D object
        # Store information that is enough to recover the detection
        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : [class_name],                         # global class id for this detection
            'class_id' : [global_class_id],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': class_name in BG_CLASSES,
            
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'voxel_index': voxel_index,
            'voxel_size' : voxel_size,
            'bbox': pcd_bbox,
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
        }
        
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
    
    return fg_detection_list, bg_detection_list
def gobs_to_detection_list(
    cfg, 
    image, 
    pcd_array,
    idx, 
    gobs, 
    trans_pose = None,
    class_names = None,
    BG_CLASSES = ["wall", "floor", "ceiling"],
    color_path = None,
    voxel_size = 5
):
    '''
    Return a DetectionList object from the gobs
    All object are still in the camera frame. 
    '''
    fg_detection_list = DetectionList()
    bg_detection_list = DetectionList()
    
    gobs = resize_gobs(gobs, image)
    gobs = filter_gobs(cfg, gobs, image, BG_CLASSES)
    
    if len(gobs['xyxy']) == 0:
        return fg_detection_list, bg_detection_list
    
    # Compute the containing relationship among all detections and subtract fg from bg objects
    xyxy = gobs['xyxy']
    mask = gobs['mask']
    gobs['mask'] = mask_subtract_contained(xyxy, mask)
    
    n_masks = len(gobs['xyxy'])
    for mask_idx in range(n_masks):
        mask = gobs['mask'][mask_idx]
        class_name = gobs['classes'][mask_idx]
        global_class_id = -1 if class_names is None else class_names.index(class_name)
    
        camera_object_pcd = create_object_pcd_from_pcd(
        pcd_array,
        image,
        mask,
        obj_color = None
    )
        if len(camera_object_pcd.points) < max(cfg.min_points_threshold, 5): 
            continue
        
        if trans_pose is not None:
            global_object_pcd = camera_object_pcd.transform(trans_pose)
        else:
            global_object_pcd = camera_object_pcd
        
        # get largest cluster, filter out noise 
        # global_object_pcd = process_pcd(global_object_pcd, cfg)

        voxel_index = mapPcdToVoxel(np.array(global_object_pcd.points),voxel_size=voxel_size)
        pcd_bbox = get_bounding_box(cfg, global_object_pcd)
        pcd_bbox.color = [0,1,0]
        
        if pcd_bbox.volume() < 1e-6:
            continue

        detected_object = {
            'image_idx' : [idx],                             # idx of the image
            'mask_idx' : [mask_idx],                         # idx of the mask/detection
            'color_path' : [color_path],                     # path to the RGB image
            'class_name' : [class_name],                         # global class id for this detection
            'class_id' : [global_class_id],                         # global class id for this detection
            'num_detections' : 1,                            # number of detections in this object
            'mask': [mask],
            'xyxy': [gobs['xyxy'][mask_idx]],
            'n_points': [len(global_object_pcd.points)],
            'pixel_area': [mask.sum()],
            'contain_number': [None],                          # This will be computed later
            "inst_color": np.random.rand(3),                 # A random color used for this segment instance
            'is_background': class_name in BG_CLASSES,
            
            # These are for the entire 3D object
            'pcd': global_object_pcd,
            'voxel_index': voxel_index,
            'voxel_size' : voxel_size,
            'bbox': pcd_bbox,
            'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),
            'text_ft': to_tensor(gobs['text_feats'][mask_idx]),
        }
        
        if class_name in BG_CLASSES:
            bg_detection_list.append(detected_object)
        else:
            fg_detection_list.append(detected_object)
    
    return fg_detection_list, bg_detection_list

def transform_detection_list(
    detection_list: DetectionList,
    transform: torch.Tensor,
    deepcopy = False,
):
    '''
    Transform the detection list by the given transform
    
    Args:
        detection_list: DetectionList
        transform: 4x4 torch.Tensor
        
    Returns:
        transformed_detection_list: DetectionList
    '''
    transform = to_numpy(transform)
    
    if deepcopy:
        detection_list = copy.deepcopy(detection_list)
    
    for i in range(len(detection_list)):
        detection_list[i]['pcd'] = detection_list[i]['pcd'].transform(transform)
        detection_list[i]['bbox'] = detection_list[i]['bbox'].rotate(transform[:3, :3], center=(0, 0, 0))
        detection_list[i]['bbox'] = detection_list[i]['bbox'].translate(transform[:3, 3])
        # detection_list[i]['bbox'] = detection_list[i]['pcd'].get_oriented_bounding_box(robust=True)
    
    return detection_list
def camera_param_to_camera_matrix(camera_param):
    f = camera_param.f
    cx = camera_param.xc
    cy = camera_param.zc
    camera_matrix = np.array([[f, 0.0, cx, 0.0],
                              [0.0, f, cy, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]])
    return camera_matrix
