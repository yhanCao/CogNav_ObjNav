import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms

from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.objectgoal_hm3d import ObjectGoal_Env_hm3d
from utils.constants import color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu
import networkx as nx
import matplotlib.colors as mcolors
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

class Sem_Exp_Env_Agent(ObjectGoal_Env_hm3d):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):

        self.args = args
        super().__init__(args, rank, config_env, dataset)

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID

        self.device = args.device
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None

        self.replan_count = 0
        self.collision_n = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

        self.fail_case = {}
        self.fail_case['collision'] = 0
        self.fail_case['success'] = 0
        self.fail_case['detection'] = 0
        self.fail_case['exploration'] = 0

        self.eve_angle = 0

    def reset(self):
        args = self.args

        self.replan_count = 0
        self.collision_n = 0

        obs, info = super().reset()
        _obs = obs.copy()
        obs = self._preprocess_obs(obs)

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None

        self.eve_angle = 0
        self.eve_angle_old = 0

        info['eve_angle'] = self.eve_angle
        info['ori_obs'] = _obs

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        # s_time = time.time()

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), self.fail_case, False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            goal = planner_inputs['goal']
            if np.sum(goal == 1) == 1 and self.args.task_config == "tasks/objectnav_gibson.yaml":
                frontier_loc = np.where(goal == 1)
                self.info["g_reward"] = self.get_llm_distance(planner_inputs["map_target"], frontier_loc)
            self.info['clear_flag'] = 0

        action,replan = self._plan(planner_inputs)

        if self.collision_n > 40 or self.replan_count > 20:
            self.info['clear_flag'] = 1
            self.collision_n = 0

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)

        if action >= 0:
            action = {'action': action}
            obs, rew, done, info = super().step(action)
            if planner_inputs['done'] :
                done =1 
            if done and self.info['success'] == 0:
                if self.info['time'] >= self.args.max_episode_length - 1:
                    self.fail_case['exploration'] += 1
                elif self.replan_count > 26:
                    self.fail_case['collision'] += 1
                else:
                    self.fail_case['detection'] += 1

            if done and self.info['success'] == 1:
                self.fail_case['success'] += 1
            _obs = obs.copy()
            obs = self._preprocess_obs(obs) 
            self.last_action = action['action']
            self.obs = obs
            self.info = info
            info['eve_angle'] = self.eve_angle
            info['ori_obs'] = _obs
            info['replan'] = replan


            return obs, self.fail_case, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            self.info['replan'] = replan
            return np.zeros(self.obs_shape), self.fail_case, False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])  # 四舍五入
        exp_pred = np.rint(planner_inputs['exp_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1, start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                      int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start, self.visited_vis[gx1:gx2, gy1:gy2])

        # Collision check
        if self.last_action == 1 and not planner_inputs["new_goal"]:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                self.collision_n += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                        self.collision_map[r, c] = 1
    
        stg, replan, stop = self._get_stg(map_pred, start, np.copy(goal), planning_window,planner_inputs['step']) # get short term

        if replan:
            self.replan_count += 1
            print("false: ", self.replan_count)
        else:
            self.replan_count = 0

        # Deterministic Local Policy
        if (stop and planner_inputs['found_goal'] == 1) or self.replan_count > 26:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            ## add the evelution angle
            eve_start_x = int(5 * math.sin(angle_st_goal) + start[0])
            eve_start_y = int(5 * math.cos(angle_st_goal) + start[1])
            if eve_start_x > map_pred.shape[0]: eve_start_x = map_pred.shape[0] 
            if eve_start_y > map_pred.shape[0]: eve_start_y = map_pred.shape[0] 
            if eve_start_x < 0: eve_start_x = 0 
            if eve_start_y < 0: eve_start_y = 0 
            if exp_pred[eve_start_x, eve_start_y] == 0 and self.eve_angle > -60:
                action = 5
                self.eve_angle -= 30
            elif exp_pred[eve_start_x, eve_start_y] == 1 and self.eve_angle < 0:
                action = 4
                self.eve_angle += 30
            elif relative_angle > self.args.turn_angle / 2.:
                action = 3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 2  # Left
            else:
                action = 1  # Forward
        return action,replan

    def _get_stg(self, grid, start, goal, planning_window,step):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], self.kernel) == 1] = 1
        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal,step)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), replan, stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs)
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        self.rgb_vis = self.rgb_vis = rgb[:, :, ::-1]
       
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            # sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        mask2 = depth > 0.99
        mask1 = depth == 0
        depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0
        depth[mask1] = 0.
        depth[mask2] = 0.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
    
        semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
        self.rgb_vis = rgb[:, :, ::-1]
        return None, semantic_pred

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/{}/{}/".format(args.dump_location, args.scenes[0],args.skip_times)
        ep_dir = '{}/episodes/'.format(
            dump_dir)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)


        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        graph = inputs['graph']
        # map_edge = inputs['map_edge']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        height, width = sem_map.shape
        sem_map_vis = np.zeros((height, width, 3), dtype=np.uint8)

        no_cat_mask = np.rint(sem_map) == -1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        # edge_mask = map_edge == 1

        # sem_map[no_cat_mask] = 0
        sem_map_vis[no_cat_mask]=np.array(color_palette[0:3])*255
        # m1 = np.logical_and(no_cat_mask, exp_mask)
        # sem_map[m1] = 2
        sem_map_vis[exp_mask]=np.array(color_palette[6:9])*255

        sem_map_vis[vis_mask]=[0,165,255]
        max_index = np.max(sem_map)
        wall_mask = np.rint(sem_map) == 1
        sem_map_vis[wall_mask]=np.array(color_palette[3:6])*255
        for i in range(2,int(max_index)+1) :
            sem_map_vis[np.rint(sem_map) == i]=np.array(color_proposals[(i-2)%len(color_proposals)]) * 255
        if graph is not None :
            all_edges_with_data = graph.edges(data=False)
            pos = nx.get_node_attributes(graph, 'pos')
            edge_nodes=[]
            node = []
            points_2d=[]
            relation=[]
            for i,pose in pos.items():
                    points_2d.append([pose[0],pose[1]])
                    node.append(i)
            for edge in all_edges_with_data:
                relation.append([node.index(edge[0]),node.index(edge[1])])
            edge_nodes = list(set(edge_nodes))
            for edge in relation:
                pt1 = (int(points_2d[edge[0]][0]),int(points_2d[edge[0]][1])) 
                pt2 = (int(points_2d[edge[1]][0]),int(points_2d[edge[1]][1]))  
                cv2.line(sem_map_vis, pt1, pt2, (255, 0, 0), 2)  
        if len(np.where(goal==1)[1]) ==1 :
            cv2.circle(sem_map_vis, np.array([np.where(goal==1)[1][0],np.where(goal==1)[0][0]]), 5, (0, 0, 255), -1)  # 红色点
        else :
            sem_map_vis[np.where(goal==1)[0],np.where(goal==1)[1]]=[0,0,255]
        sem_map_vis = np.flipud(sem_map_vis)
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
    
        
        semantic = cv2.imread(inputs['semantic_path'],cv2.IMREAD_COLOR)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1310] = semantic
        self.vis_image[50:530,1325:1805] = sem_map_vis

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )
    
        agent_arrow = vu.get_contour_points(pos, origin=(1325, 50), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            fn = '{}/episodes/{}-{}-Vis-{}.png'.format(
                dump_dir,
                self.rank, self.episode_no, self.timestep)
            cv2.imwrite(fn, self.vis_image)


