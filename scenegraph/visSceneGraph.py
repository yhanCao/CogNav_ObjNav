import copy
import json
import os
import pickle
import gzip
import argparse

import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import time
import torch.nn.functional as F
import open_clip
import distinctipy
from slam.slam_classes import MapObjectList
from slam.utils import selectPcdFromVoxel
from utils.vis import LineMesh
from gradslam.structures.pointclouds import Pointclouds
import matplotlib.colors as mcolors
css4_colors = mcolors.XKCD_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]
tt4_colors = mcolors.TABLEAU_COLORS
color_proposals2 = [list(mcolors.hex2color(color)) for color in css4_colors.values()]
def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere
def drawScenegraph(objects,edges,name=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=100)

    # Load edge files and create meshes for the scene graph
    
    colors = [color_proposals[c] for c in range(len(objects))]
    obj_centers = []
    scene_graph_geometries = []
    for i in range(len(objects)):
        pcd = selectPcdFromVoxel(objects[i])
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        # radius = extent ** 0.5 / 25
        radius = 8
        obj_centers.append(center)
        # remove the nodes on the ceiling, for better visualization
        ball = create_ball_mesh(center, radius, colors[i])

        pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
        pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors[i])
        # pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors[i])
        # o3d.visualization.draw_geometries([pcd])
        vis.add_geometry(pcd)
        scene_graph_geometries.append(ball)
    # for i in range(len(bg_objects)):
    #     pcd = selectPcdFromVoxel(bg_objects[i])
    #     points = np.asarray(pcd.points)
    #     center = np.mean(points, axis=0)
    #     # radius = extent ** 0.5 / 25
    #     # radius = 8
    #     # obj_centers.append(center)
    #     # # remove the nodes on the ceiling, for better visualization
    #     # ball = create_ball_mesh(center, radius, colors[i])

    #     pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
    #     # pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors[i])
    #     # o3d.visualization.draw_geometries([pcd])
    #     vis.add_geometry(pcd)
        # scene_graph_geometries.append(ball)
    # vis.capture_screen_image("connection.jpg")
    for id1,value in edges.items():
        for id2 in value.keys():
            if id2 in edges.keys():
                if id1 in edges[id2].keys():
                    if value[id2] =="next to" or edges[id2][id1] == 'next to':
                        line_mesh = LineMesh(
                            points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                            lines = np.array([[0, 1]]),
                            colors = [118/255,218/255,145/255],
                            radius=3
                        )
                    elif value[id2] == 'on' or value[id2] == 'under' or edges[id2][id1] == 'on' or edges[id2][id1] == 'under':
                        line_mesh = LineMesh(
                            points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                            lines = np.array([[0, 1]]),
                            colors = [250/255,109/255,29/255],
                            radius=3
                        )
                    elif value[id2] == 'in' or edges[id2][id1] == 'in':
                        line_mesh = LineMesh(
                            points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                            lines = np.array([[0, 1]]),
                            colors = [248/255,203/255,127/255],
                            radius=3
                        )
                    elif value[id2] == 'hanging on' or  edges[id2][id1] == 'hanging on':
                        
                        line_mesh = LineMesh(
                            points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                            lines = np.array([[0, 1]]),
                            colors = [99/255,178/255,238/255],
                            radius=3
                        )
            else :
                if value[id2] == 'on' or value[id2] == 'under':
                    line_mesh = LineMesh(
                        points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                        lines = np.array([[0, 1]]),
                        colors = [118/255,218/255,145/255],
                        radius=3
                    )
                elif value[id2] =="next to" :
                    line_mesh = LineMesh(
                        points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                        lines = np.array([[0, 1]]),
                        colors = [250/255,109/255,29/255],
                        radius=3
                    )
                elif value[id2] == 'in':
                    line_mesh = LineMesh(
                        points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                        lines = np.array([[0, 1]]),
                        colors = [248/255,203/255,127/255],
                        radius=3
                    )
                elif value[id2] == 'hanging on':
                    
                    line_mesh = LineMesh(
                        points = np.array([obj_centers[int(id1)], obj_centers[int(id2)]]),
                        lines = np.array([[0, 1]]),
                        colors = [99/255,178/255,238/255],
                        radius=3
                    )
            scene_graph_geometries.extend(line_mesh.cylinder_segments)
            
        for geometry in scene_graph_geometries:
            vis.add_geometry(geometry, reset_bounding_box=False)
    # vis.capture_screen_image("connection.jpg")
    vis.run()
    # vis.poll_events()
    # vis.update_renderer()
    # time.sleep(2)
    # if name is not None :
    #     # vis.update_renderer()
    #     vis.capture_screen_image(name)
    vis.destroy_window()
def draw_Voronoi(objects,bg_objects,floor_pcd,boundarypcd,lineset):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=100)

    # Load edge files and create meshes for the scene graph
    vis.add_geometry(floor_pcd)
    colors = [color_proposals[c] for c in range(len(objects))]
    colors_bg = [color_proposals2[c] for c in range(len(objects))]
    obj_centers = []
    scene_graph_geometries = []
    for i in range(len(objects)):
        pcd = selectPcdFromVoxel(objects[i])
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        # radius = extent ** 0.5 / 25
        radius = 8
        obj_centers.append(center)
        # remove the nodes on the ceiling, for better visualization
        ball = create_ball_mesh(center, radius, colors[i])

        pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
        
        # o3d.visualization.draw_geometries([pcd])
        vis.add_geometry(pcd)
        scene_graph_geometries.append(ball)
    for i in range(len(bg_objects)):
        # print(bg_objects[i]['class_name'])
        if 'ceiling' not in bg_objects[i]['class_name']:
            pcd = selectPcdFromVoxel(bg_objects[i])
            pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
            vis.add_geometry(pcd)
    vis.add_geometry(boundarypcd)
    vis.add_geometry(lineset)
    vis.run()
    vis.destroy_window()
def drawScenegraph2(objects,bg_objects,pcd_boundary,nodes,edges,name=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="scene",width=1280,height=960,left=100)

    # Load edge files and create meshes for the scene graph
    # vis.add_geometry(floor_pcd)
    colors = [color_proposals[c] for c in range(len(objects))]
    colors_bg = [color_proposals2[c] for c in range(len(objects))]
    obj_centers = []
    scene_graph_geometries = []
    for i in range(len(objects)):
        pcd = selectPcdFromVoxel(objects[i])
        points = np.asarray(pcd.points)
        center = np.mean(points, axis=0)
        # radius = extent ** 0.5 / 25
        radius = 8
        obj_centers.append(center)
        # remove the nodes on the ceiling, for better visualization
        ball = create_ball_mesh(center, radius, colors[i])

        pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
        
        # o3d.visualization.draw_geometries([pcd])
        vis.add_geometry(pcd)
        scene_graph_geometries.append(ball)
    for i in nodes :
        vis.add_geometry(i)
    for i in range(len(bg_objects)):
        # print(bg_objects[i]['class_name'])
        if 'ceiling' not in bg_objects[i]['class_name']:

            pcd = selectPcdFromVoxel(bg_objects[i])

            pcd.normals=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points)))
            # pcd.colors=o3d.utility.Vector3dVector(np.ones_like(np.asarray(pcd.points))*colors_bg[i])
            # o3d.visualization.draw_geometries([pcd])
            vis.add_geometry(pcd)
    vis.add_geometry(pcd_boundary)
    vis.add_geometry(edges)
    vis.run()
    vis.destroy_window()