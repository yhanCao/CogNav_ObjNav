from scipy.spatial import  Voronoi,voronoi_plot_2d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
import open3d as o3d
from collections import deque
import os
from scenegraph.visSceneGraph import drawScenegraph,drawScenegraph2,draw_Voronoi
import matplotlib.colors as mcolors
import cv2
from scipy.ndimage import binary_erosion
from scipy.spatial._qhull import QhullError
from envs.utils.fmm_planner import FMMPlanner
from skimage.draw import line
from tqdm import tqdm
import time
import math
from networkx.exception import NetworkXNoPath
import copy
css4_colors = mcolors.TABLEAU_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]
def VorRemoveOut(vor,map_2d,obs_map):
    vertices=vor.vertices
    relation = vor.ridge_vertices
    b_f,b_u = np.min(vor.points,axis=0),np.max(vor.points,axis=0)
    index1 = np.where(((vertices >= b_f) & (vertices < b_u))[:,0] & ((vertices >= b_f) & (vertices < b_u))[:,1])[0]
    verticesfloor= np.round(vertices[index1]).astype(np.int32)
    index_final = np.where(map_2d[verticesfloor[:,1],verticesfloor[:,0]]==1)[0]
    index_remain = index1[index_final]
    v_remain = vertices[index_remain]
    points =np.vstack((np.where(obs_map==1)[1],np.where(obs_map==1)[0])).T
    # print(points.shape)
    distance = np.linalg.norm(v_remain[:,None,:] - points[None,:,:],axis=-1)
    index_remain= list(set(index_remain)-set(index_remain[np.where(distance < 4)[0]]))
    vor.vertices = vertices[index_remain]
    relation_new=[]
    for i in range(len(relation)):
        if relation[i][0] in index_remain and relation[i][1] in index_remain and relation[i][0] >=0 and relation[i][1] >=0:
            relation_new.append((index_remain.index(relation[i][0]),index_remain.index(relation[i][1])))
    vor.ridge_vertices = relation_new
    vor.vertices= vertices[index_remain]
    return vor
def visual2Dgraph(graph,step,n,save_path):
    pos = nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color='orange', edge_color='red')
    plt.title('Graph with Largest Connected Component Highlighted')
    plt.savefig(save_path+str(step)+"_"+str(n)+".png")
    plt.close() 
def remove_isolated_nodes(G):
    isolated_nodes = [node for node, degree in G.degree() if degree == 0]
    G.remove_nodes_from(isolated_nodes)
    return G
def judgePassObstacle(pos1,pos2,obstacle_map):
    rr, cc = line(int(pos1[1]), int(pos1[0]), int(pos2[1]), int(pos2[0]))
    passed_occupied = np.any(obstacle_map[rr, cc] == 1)
    if passed_occupied:
        return False
    else :
        return True
def calculate_angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  
    angle_deg = np.degrees(angle_rad)  
    return angle_deg
def simplify_graph(G):
    H = G.copy()
    nodes_to_remove = [node for node in H.nodes() if H.degree(node) == 2]
    
    for node in nodes_to_remove:
       while H.degree(node) == 2:  
            neighbors = list(H.neighbors(node))
            if len(neighbors) == 2:
                H.add_edge(neighbors[0], neighbors[1])
                H.remove_edge(node, neighbors[0])
                H.remove_edge(node, neighbors[1])
    H=remove_isolated_nodes(H)
    return H
def simplify_graph2(G):
    H = G.copy()
    pos = nx.get_node_attributes(H, 'pos')
    nodes_to_remove = [node for node in H.nodes() if H.degree(node) == 2]
    for node in nodes_to_remove:
       while H.degree(node) == 2:  
            neighbors = list(H.neighbors(node))
            if len(neighbors) == 2:
                vector1 = np.array(pos[neighbors[0]]) - np.array(pos[node])
                vector2 = np.array(pos[neighbors[1]]) - np.array(pos[node])
                angle = calculate_angle_between_vectors(vector1, vector2)
                if angle > 150 :
                    H.add_edge(neighbors[0], neighbors[1])
                    H.remove_edge(node, neighbors[0])
                    H.remove_edge(node, neighbors[1])
                if angle <= 150 :
                    break                        
    H=remove_isolated_nodes(H)
    return H

def getAngle(G,node1,node2):
    neighbor1 = list(G.neighbors(node1))[0]
    neighbor2 = list(G.neighbors(node2))[0]
    a1 = np.array([G.nodes()[node1]['pos'][0]-G.nodes()[neighbor1]['pos'][0],G.nodes()[node1]['pos'][1]-G.nodes()[neighbor1]['pos'][1]])
    a2 = np.array([G.nodes()[node2]['pos'][0]-G.nodes()[neighbor2]['pos'][0],G.nodes()[node2]['pos'][1]-G.nodes()[neighbor2]['pos'][1]])
    if np.dot(a1,a2)/(np.linalg.norm(a1)*np.linalg.norm(a2)) > np.cos(np.radians(45)) :
        return True
    else :
        return False

def getLeafValue(map_2d,graph):
    leaf_nodes = [node for node in graph.nodes if graph.degree[node] == 1 and 'pos' in graph.nodes[node].keys()]
    node = []
    points_2d=[]
    direction = []
    for i in leaf_nodes:
        points_2d.append([graph.nodes()[i]['pos'][0],graph.nodes()[i]['pos'][1]])
        neighbor =  list(graph.neighbors(i))[0]
        direction.append([graph.nodes()[i]['pos'][0]-graph.nodes()[neighbor]['pos'][0],graph.nodes()[i]['pos'][1]-graph.nodes()[neighbor]['pos'][1]])
        node.append(i)
    
    direction = np.array(direction)
    points_2d = np.array(points_2d)
    direction = direction/np.linalg.norm(direction,axis=-1,keepdims=True)
    scales = np.arange(1,11).reshape(10,1)
    direction = direction[:, np.newaxis, :] * scales
    points_index = direction + points_2d[:,np.newaxis,:]
    indices = np.floor(np.array(points_index)).astype(np.int16)
    indices = np.clip(indices,np.array([0,0]),np.array([map_2d.shape[1]-1,map_2d.shape[0]-1]))
    value = map_2d[indices.reshape(-1,2)[:,1],indices.reshape(-1,2)[:,0]].reshape(-1,10)
    non_zero_one_mask = ((value != 0) * (value != 1))
    first_non_zero_one_indices = np.argmax(non_zero_one_mask, axis=1)
    first_non_zero_one_indices[np.all(~non_zero_one_mask, axis=1)] = -1
    first_non_zero_values = value[np.arange(value.shape[0]), first_non_zero_one_indices]
    # first_non_zero_values[first_non_zero_one_indices == -1] = 0
    leaf_value = first_non_zero_values
    value_dict={}
    value_position={}
    # remove_ele = []
    for node,value in zip(leaf_nodes,leaf_value):
        # if value !=0 :
            value_dict[node] = value
            value_position[node] = graph.nodes()[node]['pos']
    return value_dict,value_position

def mergeGraphByObjects(graph,value_dict,leaf_position):

    nearest_leaf_nodes={}
    # start = time.time()
    values = np.array(list(value_dict.values()))
    # print("leaf_values:",values)
    keys = np.array(list(value_dict.keys()))
    value_matrix = (values[:, None] == values[None, :]).astype(int)
    np.fill_diagonal(value_matrix, 0)
    positions= np.array(list(leaf_position.values()))
    distance = np.linalg.norm(positions[None,:,:]-positions[:,None,:],axis=-1)
    # print(value_matrix.shape,distance.shape)
    judge = ((distance <= 20) *(value_matrix==1))
    judge[np.tril_indices_from(judge, k=-1)] = 0
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    for node in range(len(judge)):
        nearest_leaf = []
        related_node = keys[np.where(judge[node])[0]]
        # print(keys,related_node)
        for node_r in related_node :
            if keys[node] in shortest_path_lengths.keys() and node_r in shortest_path_lengths[keys[node]].keys() and shortest_path_lengths[keys[node]][node_r] <= 5 :
                if getAngle(graph,keys[node],node_r):
                    nearest_leaf.append(node_r)
        nearest_leaf_nodes[keys[node]]=nearest_leaf

    group_dict = {}
    group = []
    for key,values in nearest_leaf_nodes.items():
        if len(values) == 0 :
            length=len(group)
            group.append([key])
            group_dict[key]=length
        else :
            for value in values :
                if key in group_dict.keys():
                    if value not in group_dict.keys() :
                        group_dict[value] = group_dict[key]
                        group[group_dict[key]].append(value)
                else :
                    if value in group_dict.keys():
                        group_dict[key] = group_dict[value]
                        group[group_dict[value]].append(key)
                    else :
                        length=len(group)
                        group.append([key,value])
                        group_dict[key]=length
                        group_dict[value]=length
    return group

def CoarseGraph(graph,group,leaf_nodes):
    # print(group)
    G = graph.copy()
    pos = nx.get_node_attributes(G, 'pos')
    for grp in group :
        if len(grp) == 2 :
            G.remove_node(grp[1])
            del leaf_nodes[grp[1]]
        elif len(grp) > 2 :
            node_ave=np.array([0,0],dtype=np.float64)
            nodes_array=[]
            neighbors=[]
            for i in grp :
                node_ave += np.array(pos[i])
                nodes_array.append(list(pos[i]))
                neighbors.append(list(graph.neighbors(i))[0])
            node_ave /= len(grp)
            merged_to_node = np.argmin(np.linalg.norm(node_ave-np.array(nodes_array),axis=-1))
            for i,n in zip(grp,neighbors) :
                if i != grp[merged_to_node] :
                    # if i in leaf_nodes :
                        del leaf_nodes[i]
                        G.remove_node(i)
    return G,leaf_nodes

def visualizationVoro(objects,bg_objects,floor_pcd,voronoi,bbox):
    points_2d=(voronoi.points+np.array([bbox[0]-1,bbox[1]-1])+0.5)*5
    vertices_2d = (voronoi.vertices+np.array([bbox[0]-1,bbox[1]-1])+0.5)*5
    relation=voronoi.ridge_vertices
    points_3d = np.append(points_2d,np.ones((points_2d.shape[0],1)),1)
    vertices_3d = np.append(vertices_2d,np.ones((vertices_2d.shape[0],1)),1)
    colors_p = np.ones_like(points_3d)*[1,0,0]
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points_3d)
    pcd.normals=o3d.utility.Vector3dVector(points_3d)
    pcd.colors=o3d.utility.Vector3dVector(colors_p)
    colors = [[0, 0, 1] for i in range(len(relation))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices_3d),
        lines=o3d.utility.Vector2iVector(np.array(relation)),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    draw_Voronoi(objects,bg_objects,floor_pcd,pcd,line_set)

def euclidean_distance(pos, node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def merge_closest_non_leaf_nodes(graph,min_distance = 5):
    non_leaf_nodes = list(nx.get_node_attributes(graph, 'pos').keys())
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    # while len(non_leaf_nodes) > 1:
    closest_pair = []
    pos = nx.get_node_attributes(graph, 'pos')
    for i, node1 in enumerate(non_leaf_nodes):
        
        for node2 in non_leaf_nodes[i + 1:]:
            distance = euclidean_distance(pos[node1], pos[node2])
            if distance < min_distance and node1 in lengths.keys() and node2 in lengths[node1].keys() and lengths[node1][node2] < 3:
                closest_pair.append((node1, node2)) 
        if closest_pair:
            for pair in closest_pair:
                node1, node2 = pair
                if graph.has_node(node1) and graph.has_node(node2) :
                    num_subnodes1 = graph.degree(node1)
                    num_subnodes2 = graph.degree(node2)
                    if num_subnodes1 >= num_subnodes2 :
                        graph = nx.contracted_nodes(graph, node1, node2, self_loops=False)
                    else :
                        graph = nx.contracted_nodes(graph, node2, node1, self_loops=False)

    return graph


def compute_euclidean_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
def judgeDirection(G1,G2,node1,node2):
    neighbor1 = list(G1.neighbors(node1))[0]
    neighbor2 = list(G2.neighbors(node2))[0]
    a1 = np.array([G1.nodes()[node1]['pos'][0]-G1.nodes()[neighbor1]['pos'][0],G1.nodes()[node1]['pos'][1]-G1.nodes()[neighbor1]['pos'][1]])
    a2 = np.array([G2.nodes()[node2]['pos'][0]-G2.nodes()[neighbor2]['pos'][0],G2.nodes()[node2]['pos'][1]-G2.nodes()[neighbor2]['pos'][1]])

    dot_product = np.dot(a1, -a2)
    
    norm_v1 = np.linalg.norm(a1)
    norm_v2 = np.linalg.norm(a2)
    
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    cos_30 = np.cos(np.radians(30))
    
    return cos_theta > cos_30



def sparsify_graph(floor_graph: nx.Graph, resampling_dist: float = 10):
        """
        Sparsify a topology graph by removing nodes with degree 2.
        This algorithm first starts at degree-one nodes (dead ends) and
        removes all degree-two nodes until confluence nodes are found.
        Next, we find close pairs of higher-order degree nodes and
        delete all nodes if the shortest path between two nodes consists
        only of degree-two nodes.
        Args:
            floor_graph (nx.Graph): graph to sparsify
        Returns:
            nx.Graph: sparsified graph
        """
        graph = copy.deepcopy(floor_graph)

        if len(graph.nodes) < 10:
            return graph
        # all nodes with degree 1 or 3+
        new_node_candidates = [
            node for node in list(graph.nodes) if (graph.degree(node) != 2)
        ]

        new_graph = nx.Graph()
        for i, node in enumerate(new_node_candidates):
            new_graph.add_node(
                node,
                pos=graph.nodes[node]["pos"]
            )
        new_nodes = set(new_graph.nodes)
        new_nodes_list = list(new_graph.nodes)

        print(
            f"Getting paths between all nodes. Node number: {len(new_node_candidates)}/{len(graph.nodes)}"
        )

        st = time.time()
        all_path_dense_graph = dict(nx.all_pairs_dijkstra_path(graph, weight="dist"))
        ed = time.time()
        print("time for computing all pairs shortest path: ", ed - st, " seconds")
        sampled_edges_to_add = list()
        pbar = tqdm(range(len(new_graph.nodes)), desc="Sparsifying graph")
        for i in pbar:
            inner_pbar = tqdm(
                range(len(new_graph.nodes)), desc="Sparsifying graph", leave=False
            )
            for j in inner_pbar:
                if i >= j:
                    continue
                # Go through all edges along path and extract dist
                node1 = new_nodes_list[i]
                node2 = new_nodes_list[j]
                try:
                    path = all_path_dense_graph[node1][node2]
                    for node in path[1:-1]:
                        if graph.degree(node) > 2:
                            break
                    else:
                        sampled_edges_to_add.append(
                            (
                                path[0],
                                path[-1],
                                np.linalg.norm(np.array(path[0]) - np.array(path[-1])),
                            )
                        )
                        dist = [
                            graph.edges[path[k], path[k + 1]]["dist"]
                            for k in range(len(path) - 1)
                        ]
                        mov_agg_dist = 0
                        predecessor = path[0]
                        # connect the nodes if there is a path between them that does not go through any other of the new nodes
                        if (
                            len(path)
                            and len(set(path[1:-1]).intersection(new_nodes)) == 0
                        ):
                            for cand_idx, cand_node in enumerate(path[1:-1]):
                                mov_agg_dist += dist[cand_idx]
                                print(mov_agg_dist)
                                if mov_agg_dist > resampling_dist:
                                    sampled_edges_to_add.append(
                                        (
                                            predecessor,
                                            cand_node,
                                            np.linalg.norm(
                                                np.array(predecessor)
                                                - np.array(cand_node)
                                            ),
                                        )
                                    )
                                    predecessor = cand_node
                                    mov_agg_dist = 0
                                else:
                                    continue
                            sampled_edges_to_add.append(
                                (
                                    predecessor,
                                    path[-1],
                                    np.linalg.norm(
                                        np.array(predecessor) - np.array(path[-1])
                                    ),
                                )
                            )
                except:
                    continue

        for edge_param in sampled_edges_to_add:
            k, l, dist = edge_param
            if k not in new_graph.nodes:
                new_graph.add_node(
                    k, pos=graph.nodes[k]["pos"]
                )
            if l not in new_graph.nodes:
                new_graph.add_node(
                    l, pos=graph.nodes[l]["pos"]
                )
            new_graph.add_edge(k, l, dist=dist)

        return new_graph
def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def add_new_nodes_with_condition2(G, new_nodes):
    # 遍历新节点
    pos = nx.get_node_attributes(G, 'pos') 
    explored_nodes=[]
    n=len(pos)
    for new_coord in new_nodes:
        # 计算新节点与图中所有现有节点的最小距离
        min_distance = float('inf')
        closest_node = None
        
        for existing_node in G.nodes(data=True):
            existing_coord = existing_node[1]['pos']
            distance = euclidean_distance(new_coord, existing_coord)
            
            if distance < min_distance:
                min_distance = distance
                closest_node = existing_node[0]
        
        # 如果最小距离大于10，添加新节点并连接
        if min_distance > 20:
            G.add_node(n, pos=new_coord)  # 添加新节点到图中
            if G.degree(closest_node) == 1 :
                closest_node = list(G.neighbors(closest_node))[0]
            G.add_edge(n, closest_node)  # 连接到距离最小的节点
            explored_nodes.append(n)
            n=n+1
        else :
            explored_nodes.append(closest_node)
    return G,list(set(explored_nodes))
def add_new_nodes_with_condition(G, new_nodes):
    pos = nx.get_node_attributes(G, 'pos') 
    explored_nodes=[]
    n=len(pos)
    for new_coord in new_nodes:

        min_distance = float('inf')
        closest_node = None
        
        for existing_node in G.nodes(data=True):
            existing_coord = existing_node[1]['pos']
            distance = euclidean_distance(new_coord, existing_coord)
            
            if distance < min_distance:
                min_distance = distance
                closest_node = existing_node[0]
    
        explored_nodes.append(closest_node)
    return G,list(set(explored_nodes))
def getFrontierNode(map,full_w,full_h,save_path,step):
    ex = np.zeros((full_w, full_h))
    kernel = np.ones((5, 5), dtype=np.uint8)
    _local_ob_map = map[0][0].cpu().numpy()
    local = cv2.dilate(_local_ob_map, kernel)

    show_ex = cv2.inRange(map[0][1].cpu().numpy(), 0.1, 1)

    
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contour, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contour) > 0:
            contour = max(contour, key=cv2.contourArea)
            cv2.drawContours(ex, contour, -1, 1, 1) #  会原地修改输入图像
    target_edge = ex-local
    target_edge[target_edge>0.8] = 1.0
    target_edge[target_edge!=1.0] = 0.0
    target_edge = target_edge *255
    frontiers=[]
    target_edge = cv2.convertScaleAbs(target_edge)
    target_edge = cv2.cvtColor(target_edge, cv2.COLOR_GRAY2BGR)
    from skimage import measure
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True) # 输出二值图像中所有连通域
    props = measure.regionprops(img_label) # 输出连通域的属性，包括面积等
    for i in props:
        if i.area > 12 :
            cv2.circle(target_edge,(int(i.centroid[1]),int(i.centroid[0])),10,(0,255,0),2)
            frontiers.append((int(i.centroid[1]),int(i.centroid[0])))
    cv2.imwrite(save_path+"frontier"+str(step)+".png",target_edge)
    return frontiers
def generateVoronoi(map,history_node,full_w,full_h,step,save_path):
    save_map_path = os.path.join(save_path,"map/")
    if not os.path.exists(save_map_path):
        os.makedirs(save_map_path)
    frontier_nodes_2D = getFrontierNode(map,full_w,full_h,save_map_path,step)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ex_map = map[0,1].cpu().numpy()
    # ex_map = cv2.dilate(ex_map, kernel)
    node = np.vstack(history_node)
    ex_map[np.vstack([node,node-1,node+1,node+2])[:,1],np.vstack([node,node-1,node+1,node+2])[:,0]]=1
    # ex_map = cv2.medianBlur(ex_map, 5)
    free_map = cv2.morphologyEx(ex_map, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cv2.imwrite(save_map_path+"free_map1_"+str(step)+".png",free_map*255)
    obs_map = map[0,0].cpu().numpy()
    obstacle_map = map[0,4].cpu().numpy()
    obs_map = cv2.medianBlur(obs_map, 3)
    obs_map = cv2.dilate(obs_map, kernel)
    free_map[0:2, 0:full_h] = 0.0
    free_map[full_w-2:full_w, 0:full_h-1] = 0.0
    free_map[0:full_w, 0:2] = 0.0
    free_map[0:full_w, full_h-2:full_h] = 0.0
    
    cv2.imwrite(save_map_path+"obs_map_"+str(step)+".png",obs_map*255)
    path_map = free_map - obs_map
    cv2.imwrite(save_map_path+"path_map_"+str(step)+".png",path_map*255)
    boundary_map = binary_erosion(path_map, iterations=1).astype(np.uint8)
    boundary_map = path_map - boundary_map
    rows, cols = np.where(boundary_map == 1)
    boundaries = np.array(list(zip(cols, rows)))
    try:
        vor = Voronoi(boundaries) 
    except QhullError as e :
        return None,None,ex_map,obs_map,None,None
    vor=VorRemoveOut(vor,path_map,obs_map)
    if len(vor.ridge_vertices) < 1 :
        return None,None,ex_map,obs_map,None,None
    G = nx.Graph()
    for i, point in enumerate(vor.vertices):
        G.add_node(i, pos=(point[0],point[1]))
    G.add_edges_from(vor.ridge_vertices)
    all_edges_with_data = G.edges()
    pos = nx.get_node_attributes(G, 'pos') 
    remove_edge = []
    for edge in all_edges_with_data:
        if judgePassObstacle(pos[edge[0]],pos[edge[1]],obs_map) == False:
            remove_edge.append(edge)
    for edge in remove_edge :
        G.remove_edge(edge[0],edge[1])
    visual2Dgraph(G,step,0,save_map_path)
    subgraph = simplify_graph(G)
    visual2Dgraph(subgraph,step,1,save_map_path)
    if len(subgraph.nodes()) < 1 :
        return None,None,ex_map,obs_map,None,None
    leaf_values,leaf_position = getLeafValue(obstacle_map,subgraph)
    group = mergeGraphByObjects(subgraph,leaf_values,leaf_position)
    subgraph,leaf_values = CoarseGraph(subgraph,group,leaf_values)
    subgraph = simplify_graph2(subgraph)
    visual2Dgraph(subgraph,step,2,save_map_path)
    pos = nx.get_node_attributes(subgraph, 'pos') 
    deleted=[]
    for node in subgraph.nodes() :
        if node in leaf_values.keys() and leaf_values[node] != -1 and euclidean_distance(pos[node],pos[list(subgraph.neighbors(node))[0]]) <= 15:
            deleted.append(node)
    for d in deleted :
        subgraph.remove_node(d)
    subgraph = merge_closest_non_leaf_nodes(subgraph,10)
    visual2Dgraph(subgraph,step,3,save_map_path)
    if len(subgraph.nodes()) < 1 :
        return None,None,ex_map,obs_map,None,None
    subgraph=remove_isolated_nodes(subgraph)
    mapping = {old_label: new_label for new_label, old_label in enumerate(subgraph.nodes())}
    subgraph = nx.relabel_nodes(subgraph, mapping)
    visual2Dgraph(subgraph,step,4,save_map_path)
    subgraph,explored_nodes = add_new_nodes_with_condition(subgraph,history_node)
    subgraph,frontier_nodes = add_new_nodes_with_condition(subgraph,frontier_nodes_2D)
    pos = nx.get_node_attributes(subgraph, 'pos') 
    if len(list(pos.values())) < 1 :
        return None,None,ex_map,obs_map,None,None
    return subgraph,leaf_values,ex_map,obs_map,explored_nodes,frontier_nodes
def projectCurrentAgentLoc(position,G):
    pos = nx.get_node_attributes(G, 'pos') 
    min_distance = float('inf')
    closest_node = None
    for existing_node in G.nodes(data=True):
        existing_coord = existing_node[1]['pos']
        distance = euclidean_distance(position, existing_coord)
        
        if distance < min_distance:
            min_distance = distance
            closest_node = existing_node[0]
    
    if min_distance > 10:
        G.add_node(len(pos), pos=position)  
        G.add_edge(len(pos), closest_node)  
        return G,len(pos)
    else :
        return G,closest_node
def direction_to_number(normal):
    directions = np.array([
    [1, 0],    # 0°
    [1, 1],    # 45°
    [0, 1],    # 90°
    [-1, 1],   # 135°
    [-1, 0],   # 180°
    [-1, -1],  # 225°
    [0, -1],   # 270°
    [1, -1]    # 315°
    ])
    mapping={
        0:0,
        1:1,
        2:2,
        3:3,
        4:4,
        5:-3,
        6:-2,
        7:-1
    }
    dot_products = np.dot(normal[None,:], directions.T)[0]/np.linalg.norm(normal)/np.linalg.norm(directions)
    direction = np.argmax(dot_products)
    return mapping[direction]
def find_another_view(voxels_center,graph,now_position):
    pos = nx.get_node_attributes(graph, 'pos')
    nodes = np.array(list(pos.keys()))
    poses = np.array(list(pos.values()))
    distance = np.linalg.norm(poses[:,None,:]-voxels_center[None,None,:],axis=-1)
    idx = np.where(distance < 60)[0]
    if len(idx) == 0 :
        node_select = np.argmin(distance)
        print("no other view")
    else :
        select_poses = poses[idx]
        iddx=np.argmax(np.max(np.linalg.norm(select_poses[:,None,:]-now_position[None,:,:],axis=-1),axis=-1))
        node_select = nodes[idx][iddx]
    return node_select
