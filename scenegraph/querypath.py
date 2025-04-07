import networkx as nx
from collections import Counter
import numpy as np
from slam.utils import extractVoxelIndex3
import time
import string
from networkx.exception import NetworkXNoPath
def euclidean_distance(pos, node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
def QueryNextSubGoal(goal_name,llm,objects,scene_graph,graph,leaf_nodes,agent_location,distance_threshold=30):
    pos = nx.get_node_attributes(graph, 'pos')
    objects_around={}
    for node,value in leaf_nodes.items() :
        if node in pos.keys():
            for i,obj in enumerate(objects) :
                coordinates = extractVoxelIndex3(obj['voxel_index'])
                min_distance=np.min(np.linalg.norm(coordinates[:,:2]-np.array([pos[node][1],pos[node][0]]),axis=-1))
                if min_distance < distance_threshold :
                    if node in objects_around.keys() :
                        objects_around[node].append(i)
                    else :
                        objects_around[node]=[i]
    USER=f"Goal: {goal_name}\n"
    USER = USER + f"Agent Location: ({str(agent_location[0])}, {str(agent_location[1])}).\n"
    for node,value in leaf_nodes.items() :
        if node in pos.keys():
            txt=f"-Description {str(node)}: "
            if value == 0 or value == 1 :
                break
            if value == -1 :
                txt += f"This node is directed to the unexplored space, located at ({str(graph.nodes()[node]['pos'][1])}, {str(graph.nodes()[node]['pos'][0])})."
            else :
                counter = Counter(objects[int(value-2)]['class_name'])
                most_common_element, count = counter.most_common(1)[0]
                txt =txt + f"This node is directed to a {most_common_element}, located at ({str(graph.nodes()[node]['pos'][1])}, {str(graph.nodes()[node]['pos'][0])})."
            if node in objects_around.keys() :
                txt += " The area contains "
                objs_around=objects_around[node]
                for obj_index in objs_around :
                    counter = Counter(objects[obj_index]['class_name'])
                    most_common_element, count = counter.most_common(1)[0]
                    txt = txt + f"a {most_common_element}"
                    txt +="; "
            if txt[-1] =='.':
                txt+=f"\n"
            else :
                txt = f"{txt[:-2]}.\n"
            USER += txt
    print(f"=========> Query:\n{USER}")
    
    while True:
        try:
            answer, reply = llm.choose_frontier(USER)
            break
        except Exception as ex: # rate limit
            print(f"[ERROR] in LLM inference =====> {ex}, sleep for 20s...")
            time.sleep(20)
            continue
    print(f"=========> LLM output:\n{reply}")
    selection = np.array([int(graph.nodes()[answer]['pos'][0]),int(graph.nodes()[answer]['pos'][1])])
    return selection
def getObjectName(obj):
    counter = Counter(obj['class_name'])
    most_common_element, count = counter.most_common(1)[0]
    return most_common_element
def calculate_detailed_direction(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1

    if x1 == x2 and y1 == y2:
        return "Close neighbor"
    elif x1 == x2:
        if y2 > y1:
            return "Directly above"
        else:
            return "Directly below"
    elif y1 == y2:
        if x2 > x1:
            return "Directly to the right"
        else:
            return "Directly to the left"
    else:
        if dx > 0 and dy > 0:
            if abs(dx) > abs(dy):
                return "Up-right leaning right"
            else:
                return "Up-right leaning up"
        elif dx > 0 and dy < 0:
            if abs(dx) > abs(dy):
                return "Down-right leaning right"
            else:
                return "Down-right leaning down"
        elif dx < 0 and dy > 0:
            if abs(dx) > abs(dy):
                return "Up-left leaning left"
            else:
                return "Up-left leaning up"
        elif dx < 0 and dy < 0:
            if abs(dx) > abs(dy):
                return "Down-left leaning left"
            else:
                return "Down-left leaning down"
def QueryByLLM(goal_name,agent_node,obstacle_map,objects,bg_objects,scene_graph,voronoi_graph,node_rooms,explored_nodes,frontier_nodes,leaf_nodes):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Agent Now : Node {agent_node}\n"
    objects_around=getObjectNearby(objects,bg_objects,voronoi_graph,obstacle_map)
    ###describe nodes ###
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        Node_descp+=f"Node {node} : \n"
        # if node in leaf_nodes.keys() :
        #     if leaf_nodes[node] == -1 :
        #         Node_descp+=f"  Frontier node: True\n"
        #     else :
        #         Node_descp+=f"  Frontier node: False\n"
        # Node_descp+=f"  Frontier node: False\n"
        if node in frontier_nodes :
            Node_descp+=f"  Frontier node: True\n"
        else :
            Node_descp+=f"  Frontier node: False\n"
        ##########
        if node != agent_node :
            Node_descp+=f"  Location(relative to {agent_node}):\n"
            direction = calculate_detailed_direction(pos[agent_node],pos[node])
            try:
                path = nx.shortest_path(voronoi_graph, source=agent_node, target=node)
                distance= euclidean_distance(pos,agent_node,path[0])
                distance += euclidean_distance(pos,path[-1],node)
                for nidx in range(len(path)-1):
                    distance+=euclidean_distance(pos,path[nidx],path[nidx+1])
                Node_descp+=f"    Direction: {direction}\n"
                Node_descp+=f"    Path: {agent_node} ->"
                for n in path[1:-1] :
                    Node_descp+=f"{n} ->"
                Node_descp+=f"{node}\n"
                Node_descp+=f"    Distance: {distance}\n"
            except NetworkXNoPath:
                Node_descp+=f"    Direction: {direction}\n"
        if node in objects_around.keys():
            Node_descp+=f"  Surrounding objects:"
            objs_around=objects_around[node]
            for obj_message in objs_around :
                Node_descp+=f" {obj_message['obj']}, "

        if node_rooms[node] != "None" and node_rooms[node] is not None :
            Node_descp+=f"  Room: {node_rooms[node]}.\n"
        else :
            Node_descp+=f"  Room: Unknown.\n"
        if node in explored_nodes :
            Node_descp+=f"  Explored: True\n"
        else :
            Node_descp+=f"  Explored: False\n"
        Node_descp+=f"\n"
    candidate = list(pos.keys())
    candidate.remove(agent_node)
    nodes=f"The nodes candidate you select: {candidate}"
    USER = USER +Node_descp+nodes
    return USER,candidate
def LandmarkSelection_BS(goal_name,agent_node,obstacle_map,objects,bg_objects,scene_graph,voronoi_graph,node_rooms,explored_nodes,frontier_nodes,leaf_nodes):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Agent Now : Node {agent_node}\n"
    objects_around=getObjectNearby(objects,bg_objects,voronoi_graph,obstacle_map)
    ###describe nodes ###
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        Node_descp+=f"Node {node} : \n"
        if node in frontier_nodes :
            Node_descp+=f"  Frontier node: True\n"
            if node != agent_node :
                Node_descp+=f"  Location:\n"
                direction = calculate_detailed_direction(pos[agent_node],pos[node])
                try:
                    path = nx.shortest_path(voronoi_graph, source=agent_node, target=node)
                    distance= euclidean_distance(pos,agent_node,path[0])
                    distance += euclidean_distance(pos,path[-1],node)
                    for nidx in range(len(path)-1):
                        distance+=euclidean_distance(pos,path[nidx],path[nidx+1])
                    Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
                    Node_descp+=f"    Path from {agent_node} to {node}: {agent_node} ->"
                    for n in path[1:-1] :
                        Node_descp+=f"{n} ->"
                    Node_descp+=f"{node}\n"
                    Node_descp+=f"    Distance from {agent_node} to {node}: {distance}\n"
                except NetworkXNoPath:
                    Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
            if node in objects_around.keys():
                Node_descp+=f"  Surrounding objects:\n"
                objs_around=objects_around[node]
                for obj_message in objs_around :
                    Node_descp+=f"    There is {obj_message['obj']} is in direction: {obj_message['direction']}, distance: {obj_message['distance']} relativate to this node"
                    for obj,relation in scene_graph[obj_message['obj']].items():
                        Node_descp+=f" {relation} a {obj};"
                Node_descp=Node_descp[:-1]+".\n"
            if node_rooms[node] != "None" and node_rooms[node] is not None :
                Node_descp+=f"  Room: {node_rooms[node]}.\n"
            else :
                Node_descp+=f"  Room: Unknown.\n"
            if node in explored_nodes :
                Node_descp+=f"  Explored: True\n"
            else :
                Node_descp+=f"  Explored: False\n"
            Node_descp+=f"\n"
    nodes=f"The nodes candidate you select: {frontier_nodes}"
    USER = USER +Node_descp+nodes
    return USER,frontier_nodes
### remain change
def LandmarkSelection_CS(goal_name,agent_node,obstacle_map,objects,bg_objects,scene_graph,voronoi_graph,node_rooms,explored_nodes,frontier_nodes,relative_object,relative_room):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Agent Now : Node {agent_node}\n"
    objects_around=getObjectNearby(objects,bg_objects,voronoi_graph,obstacle_map)
    ###describe nodes ###
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        if node in objects_around.keys():
            if (relative_object is not None and relative_object in objects_around[node]) or (relative_room is not None and node_rooms[node] is not None and relative_room == node_rooms[node]):
                Node_descp+=f"Node {node} : \n"
                if node in frontier_nodes :
                    Node_descp+=f"  Frontier node: True\n"
                else :
                    Node_descp+=f"  Frontier node: False\n"
                ##########
                if node != agent_node :
                    Node_descp+=f"  Location:\n"
                    direction = calculate_detailed_direction(pos[agent_node],pos[node])
                    try:
                        path = nx.shortest_path(voronoi_graph, source=agent_node, target=node)
                        distance= euclidean_distance(pos,agent_node,path[0])
                        distance += euclidean_distance(pos,path[-1],node)
                        for nidx in range(len(path)-1):
                            distance+=euclidean_distance(pos,path[nidx],path[nidx+1])
                        Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
                        Node_descp+=f"    Path from {agent_node} to {node}: {agent_node} ->"
                        for n in path[1:-1] :
                            Node_descp+=f"{n} ->"
                        Node_descp+=f"{node}\n"
                        Node_descp+=f"    Distance from {agent_node} to {node}: {distance}\n"
                    except NetworkXNoPath:
                        Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
                Node_descp+=f"  Surrounding objects:\n"
                objs_around=objects_around[node]
                for obj_message in objs_around :
                    Node_descp+=f"    There is {obj_message['obj']} is in direction: {obj_message['direction']}, distance: {obj_message['distance']} relativate to this node"
                    for obj,relation in scene_graph[obj_message['obj']].items():
                        Node_descp+=f" {relation} a {obj};"
                Node_descp=Node_descp[:-1]+".\n"
                if node_rooms[node] != "None" and node_rooms[node] is not None :
                    Node_descp+=f"  Room: {node_rooms[node]}.\n"
                else :
                    Node_descp+=f"  Room: Unknown.\n"
                if node in explored_nodes :
                    Node_descp+=f"  Explored: True\n"
                else :
                    Node_descp+=f"  Explored: False\n"
                Node_descp+=f"\n"
    candidate = list(pos.keys())
    candidate.remove(agent_node)
    nodes=f"The nodes candidate you select: {candidate}"
    USER = USER +Node_descp+nodes
    return USER,candidate
def LandmarkSelection_CV(goal_name,agent_node,obstacle_map,objects,bg_objects,scene_graph,voronoi_graph,node_rooms,explored_nodes,frontier_nodes,goal_object):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Agent Now : Node {agent_node}\n"
    objects_around=getObjectNearby(objects,bg_objects,voronoi_graph,obstacle_map)
    ###describe nodes ###
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        if node in objects_around.keys():
            if (goal_object in objects_around[node]) :
                Node_descp+=f"Node {node} : \n"
                if node in frontier_nodes :
                    Node_descp+=f"  Frontier node: True\n"
                else :
                    Node_descp+=f"  Frontier node: False\n"
                ##########
                if node != agent_node :
                    Node_descp+=f"  Location:\n"
                    direction = calculate_detailed_direction(pos[agent_node],pos[node])
                    try:
                        path = nx.shortest_path(voronoi_graph, source=agent_node, target=node)
                        distance= euclidean_distance(pos,agent_node,path[0])
                        distance += euclidean_distance(pos,path[-1],node)
                        for nidx in range(len(path)-1):
                            distance+=euclidean_distance(pos,path[nidx],path[nidx+1])
                        Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
                        Node_descp+=f"    Path from {agent_node} to {node}: {agent_node} ->"
                        for n in path[1:-1] :
                            Node_descp+=f"{n} ->"
                        Node_descp+=f"{node}\n"
                        Node_descp+=f"    Distance from {agent_node} to {node}: {distance}\n"
                    except NetworkXNoPath:
                        Node_descp+=f"    Direction: {direction} relative to {agent_node}\n"
                Node_descp+=f"  Surrounding objects:\n"
                objs_around=objects_around[node]
                for obj_message in objs_around :
                    Node_descp+=f"    There is {obj_message['obj']} is in direction: {obj_message['direction']}, distance: {obj_message['distance']} relativate to this node"
                    for obj,relation in scene_graph[obj_message['obj']].items():
                        Node_descp+=f" {relation} a {obj};"
                Node_descp=Node_descp[:-1]+".\n"
                if node_rooms[node] != "None" and node_rooms[node] is not None :
                    Node_descp+=f"  Room: {node_rooms[node]}.\n"
                else :
                    Node_descp+=f"  Room: Unknown.\n"
                if node in explored_nodes :
                    Node_descp+=f"  Explored: True\n"
                else :
                    Node_descp+=f"  Explored: False\n"
                Node_descp+=f"\n"
    candidate = list(pos.keys())
    candidate.remove(agent_node)
    nodes=f"The nodes candidate you select: {candidate}"
    USER = USER +Node_descp+nodes
    return USER,candidate
def QeuryRelative(goal_name,objects,node_rooms):
    USER=f"Goal: {goal_name}\n"
    Node_descp+=f"explored_objects:"
    for obj in objects :
        obj_name = getObjectName(obj)
        Node_descp+=f" {obj_name};"
    USER+=f"\n"
    Node_descp+=f"explored_rooms:"
    room_list = set(node_rooms.values())
    for room in room_list : 
        Node_descp+=f" {room};"
    USER+=f"\n"
    return USER
def QueryState(goal_name,obstacle_map,objects,bg_objects,voronoi_graph,node_rooms,frontier_nodes,current_state,confidence):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Current State : {current_state}\n"
    objects_around=getObjectNearby(objects,bg_objects,voronoi_graph,obstacle_map)
    ###describe nodes ###
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        Node_descp+=f"Node {node} : \n"
        if node in frontier_nodes :
            Node_descp+=f"  Frontier node: True\n"
        else :
            Node_descp+=f"  Frontier node: False\n"
        if node in objects_around.keys():
            Node_descp+=f"  Surrounding objects:"
            objs_around=objects_around[node]
            for obj_message in objs_around :
                Node_descp+=f"  {obj_message['obj']};"
            Node_descp+=f"\n"
        if node_rooms[node] != "None" and node_rooms[node] is not None :
            Node_descp+=f"  Room: {node_rooms[node]}.\n"
        Node_descp+=f"\n"
    if confidence is not None :
        Node_descp+=f"The confidence of ensuring the found object is the target one : {confidence}"
    USER = USER +Node_descp
    return USER
def getObjectNearby(objects,bg_objects,graph,obstacle_map,distance_threshold=20):
    pos = nx.get_node_attributes(graph, 'pos')
    objects_around={}
    position=np.array(list(pos.values()))
    pos_id = list(pos.keys())
    edge_nodes=[]
    # 打印所有边及其属性
    obstacle_map = (obstacle_map.cpu().numpy() * 255).astype(np.uint8)
    all_edges_with_data = graph.edges(data=False)
    import cv2
    color_image1 = cv2.cvtColor(obstacle_map, cv2.COLOR_GRAY2BGR)
    node = []
    points_2d=[]
    relation=[]
    for i,pose in pos.items():
            points_2d.append([pose[0],pose[1]])
            point_int = (int(pose[0]),int(pose[1]))
            cv2.circle(color_image1, point_int, 1, (0, 0, 255), -1)  # red
            node.append(i)
    
    for edge in all_edges_with_data:
    
        relation.append([node.index(edge[0]),node.index(edge[1])])

    edge_nodes = list(set(edge_nodes))
    for edge in relation:
            pt1 = (int(points_2d[edge[0]][0]),int(points_2d[edge[0]][1]))  # 转换为整数
            pt2 = (int(points_2d[edge[1]][0]),int(points_2d[edge[1]][1]))   # 转换为整数
            cv2.line(color_image1, pt1, pt2, (255, 0, 0), 1)  # blue色线
    for obj in objects:
        obj_name = getObjectName(obj)

        coordinates = extractVoxelIndex3(obj['voxel_index'])
        # coordinates[:,0],coordinates[:,1] = coordinates[:,1],coordinates[:,0]
        center = np.mean(coordinates[:,:2],axis=0)
        point_int = (int(center[0]),int(center[1]))  # 转换为整数
        cv2.circle(color_image1, point_int, 1, (0, 0, 255), -1)  # red
        min_distance=np.min(np.linalg.norm(coordinates[None,:,:2]-position[:,None,:],axis=-1),axis=-1)
        index = np.where(min_distance < distance_threshold)[0]
        select_id = [pos_id[i] for i in index]
        for i,id in zip(index,select_id) :
            direction = calculate_detailed_direction(position[i],center)
            distance = min_distance[i]
            if id in objects_around.keys():
                objects_around[id].append({'obj':obj_name,'direction':direction,'distance':distance})
            else :
                objects_around[id]=[{'obj':obj_name,'direction':direction,'distance':distance}]
    cv2.imwrite('color_image_with_circle_and_line.png', color_image1)

    for name,bg in bg_objects.items() :
        if (name == "wall" or name == "wall-wood") and bg is not None :
            coordinates = extractVoxelIndex3(bg['voxel_index'])
            min_id=np.argmin(np.linalg.norm(coordinates[None,:,:2]-position[:,None,:],axis=-1),axis=-1)
            min_distance=np.min(np.linalg.norm(coordinates[None,:,:2]-position[:,None,:],axis=-1),axis=-1)
            index = np.where(min_distance < distance_threshold)[0]
            select_id = [pos_id[i] for i in index]
            for i,id in zip(index,select_id) :
                direction = calculate_detailed_direction(position[i],coordinates[min_id[i],:2])
                distance = min_distance[i]
                if id in objects_around.keys():
                    objects_around[id].append({'obj':'wall','direction':direction,'distance':distance})
                else :
                    objects_around[id]=[{'obj':'wall','direction':direction,'distance':distance}]
    return objects_around
def QueryByGraph(goal_name,agent_node,obstacle_map,centers,objects,bg_objects,scene_graph,voronoi_graph,node_rooms,explored_nodes,leaf_nodes):
    USER=f"Goal: {goal_name}\n"
    USER+=f"Agent Now : Node {agent_node}\n"
    uppercase_letters = list(string.ascii_uppercase)

    objects_around=getObjectNearby(objects,bg_objects,voronoi_graph,obstacle_map)
    pos = nx.get_node_attributes(voronoi_graph, 'pos')
    Node_descp=f"Node description : \n"
    for node in pos.keys():
        Node_descp+=f"Node {node} : \n"
        if node in leaf_nodes.keys() :
            if leaf_nodes[node] == -1 :
                # Node_descp+=f" a frontier node that can navigate to an unexplored space.\t"
                Node_descp+=f"  Frontier node: True\n"
            else :
                Node_descp+=f"  Frontier node: False\n"
        Node_descp+=f"  Frontier node: False\n"
        if node in objects_around.keys():
            # Node_descp += " Near this node, there are "
            Node_descp+=f"  Surrounding objects:\n"
            objs_around=objects_around[node]
            for obj_message in objs_around :
                Node_descp+=f"    There is {obj_message['obj']} is in direction: {obj_message['direction']}, distance: {obj_message['distance']} relativate to this node.\n"
        if node_rooms[node] != "None" and node_rooms[node] is not None :
            Node_descp+=f"  Room: {node_rooms[node]}.\n"
        else :
            Node_descp+=f"  Room: Unknown.\n"
        if node in explored_nodes :
            Node_descp+=f"  Explored: True\n"
        else :
            Node_descp+=f"  Explored: False\n"
        Node_descp+=f"\n"
    candidate = list(pos.keys())
    candidate.remove(agent_node)
    nodes=f"The nodes candidate you select: {candidate}"
    USER = USER +Node_descp+nodes
    return USER

