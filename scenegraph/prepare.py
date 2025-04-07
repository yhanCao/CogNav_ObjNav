from slam.utils import compute_relationship_matrix,compute_relationship_matrixreplica
import numpy as np
from utils.voronoi import merge_close_nodes
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from collections import Counter
import string
def prepareRelationForGPT(cfg,objects):
    bbox_overlaps = compute_relationship_matrixreplica(objects)
    indices = np.argwhere(bbox_overlaps == 1)
    relation_dict={}
    
    for i in range(len(bbox_overlaps)) :
        index = indices[np.where(indices[:,0]==i)[0],1].tolist()
        rr={}
        for j in index :
            rr[j]='dd'
        relation_dict[i] = rr
    return relation_dict
def prepareRelationForCOG(cfg,objects,changed,som_pth,idx):
    bbox_overlaps = compute_relationship_matrix(objects)
    indices = np.argwhere(bbox_overlaps == 1)
    relation_list=[]
    image_list =[]
    image_objects_dict={}
    image_pair_dict={}
    check_relation={}
    for i in changed :
        relation_candidate=indices[np.where(indices[:,0]==i)[0],1].tolist()
        rr={}
        for j in relation_candidate :
            if j <=i :
                break
            rr[int(j)]="dd"
            for image in list(set(objects[i]['color_path'])&set(objects[j]['color_path'])):
                if int(image.split('/')[-1].split('.')[0]) < idx-11 :
                    break
                relation_list.append([i,j])
                image_new = som_pth + image.split('/')[-1].split('.')[0]+".png"
                image_list.append(image_new)
                if image_new in image_objects_dict.keys() :
                    image_objects_dict[image_new].append(i)
                    image_objects_dict[image_new].append(j)
                    image_pair_dict[image_new].append([i,j])
                    image_objects_dict[image_new] = list(set(image_objects_dict[image_new]))
                else :
                    image_objects_dict[image_new]=[i,j]
                    image_pair_dict[image_new]=[[i,j]]
            check_relation[int(i)] = rr
    return relation_list,image_list,image_objects_dict,image_pair_dict,check_relation

def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def get_most_frequent_string_within_distance(coordinates, target_coord, max_distance):
    within_distance_strings = [
        label for coord, label in coordinates.items() 
        if euclidean_distance(coord, target_coord) <= max_distance
    ]
    if not within_distance_strings:
        return None 
    string_count = Counter(within_distance_strings)
    most_frequent_string = string_count.most_common(1)[0][0]
    return most_frequent_string


def projectHistoryToGraph(graph,current_pos,room_message,room_threshold):
    pos = nx.get_node_attributes(graph, 'pos')
    node_rooms={}
    for node,position in pos.items() :
        node_rooms[node] = get_most_frequent_string_within_distance(room_message, position, room_threshold)
    return node_rooms