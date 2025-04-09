import cv2
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from utils.common import *
import envs.utils.depth_utils as du
# from detection.groundedsam import detection, InitDetection
from slam.slam_classes import MapObjectList
from slam.mapping import removalOverlap
from utils.draw_pcd import draw_voxel_objects,draw_pcd_objects
import omegaconf
import gzip
import pickle as pkl
from scenegraph.cogvlm2 import COG_one,load_model
from omegaconf import DictConfig, OmegaConf
from slam.utils import extractVoxelCenterfromVoxelmap
from slam.cfslam import cfvoxel
import hydra
from detectron2.data import MetadataCatalog
from detection.openseed import detection,InitDetection
# from detection.groundedsam import InitDetection,detection
from detection.som import generateProjectMatrix, mapObjects_2D, dealMask
from scenegraph.prepare import prepareRelationForCOG
from detection.som import generateSomPrompt,generateSoMImage,mapObjects
import matplotlib.colors as mcolors
from detection.visualizer import Visualizer

metadata = MetadataCatalog.get('coco_2017_train_panoptic')
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

class Semantic_Mapping_SG(nn.Module):
    def __init__(self, args):
        super(Semantic_Mapping_SG, self).__init__()
        self.args = args
        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.z_resolution = args.map_resolution
        # self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.map_size_cm = args.map_size_cm
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.fov = args.hfov
        self.du_scale = args.du_scale
        self.cat_pred_threshold = args.cat_pred_threshold # 5.0
        self.exp_pred_threshold = args.exp_pred_threshold # 1.0
        self.map_pred_threshold = args.map_pred_threshold # 1.0
        self.num_sem_categories = args.num_sem_categories # 16
        self.reses_history={}
        self.selected_view=[]
        self.max_height = int(200 / self.z_resolution)
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.camera_height * 100.
        self.shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, self.fov) # 内参
        self.env_camera_matrix = du.get_camera_matrix(args.env_frame_width, args.env_frame_height, self.fov)
        self.cog_tokenizer,self.model = load_model(args.cog_model_path)
        self.num_scene = args.num_processes
        self.scene_graph={}
        self.init_grid = torch.zeros(
            self.num_scene, 1 + self.num_sem_categories, self.vision_range, self.vision_range,
            self.max_height - self.min_height
        ).float().to(self.device)

        self.feat = torch.ones(
            self.num_scene, 1 + self.num_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.stair_mask_radius = 30
        self.stair_mask = self.get_mask(self.stair_mask_radius).to(self.device)
        self.BG_CLASSES = ["wall","wall-wood", "floor", "ceiling","rug","Carpet"]
        self.cfg_detection = load_config(args.path_detection_config)
        self.cfg_detection = process_cfg(OmegaConf.create(self.cfg_detection))
        self.instance_classes_set,_ = InitDetection()

        self.global_classes = set()
        self.objects = MapObjectList(device=self.device)
        self.bg_objects = {c: None for c in self.BG_CLASSES}
        self.changed=[]
    def forward_detection(self, obs, pcd, idx, pose_curr,save_path):
        color_img_bgr = obs[:, :3].squeeze().permute(1, 2, 0)
        image_path = os.path.join(save_path, 'image')
        visualization_path = os.path.join(save_path, "visual/")
        detection_path = os.path.join(save_path, "detection/")
        pcd_path = os.path.join(save_path, "pcd/")
        pcd_save_path = os.path.join(save_path, "pcd_save/")
        som_path = os.path.join(save_path, "som/")
        object_path = os.path.join(save_path, "object/")
        
        trajectory_path = os.path.join(save_path, "trajectory/")
        visual_fig_path = os.path.join(save_path, "visual_fig/")
        visual_fig_path2 = os.path.join(save_path, "visual_fig2/")
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(detection_path):
            os.makedirs(detection_path)
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path)
        if not os.path.exists(pcd_path):
            os.makedirs(pcd_path)
        if not os.path.exists(pcd_save_path):
            os.makedirs(pcd_save_path)
        if not os.path.exists(som_path):
            os.makedirs(som_path)
        if not os.path.exists(object_path):
            os.makedirs(object_path)

        if not os.path.exists(trajectory_path):
            os.makedirs(trajectory_path)
        if not os.path.exists(visual_fig_path):
            os.makedirs(visual_fig_path)
        if not os.path.exists(visual_fig_path2):
            os.makedirs(visual_fig_path2)
        Image.fromarray(color_img_bgr.cpu().numpy().astype(np.uint8),'RGB').save(os.path.join(image_path, str(idx)+'.jpg'))
        gobs = detection(Image.fromarray(color_img_bgr.cpu().numpy().astype(np.uint8),'RGB'),idx,visualization_path,detection_path)
        # if gobs !=None :
        if gobs is not None :
            res=[]# [image; pcd; intrinsic; pose]
            res.append(color_img_bgr)
            res.append(pcd)
            res.append(camera_param_to_camera_matrix(self.env_camera_matrix))
            res.append(pose_curr)
            self.reses_history[idx]=res
            self.objects,self.bg_objects,self.changed=cfvoxel(self.cfg_detection,res,gobs,self.objects,idx=idx,step=idx,classes=self.instance_classes_set,bg_objects=self.bg_objects,img_name=os.path.join(image_path, '{}.jpg'.format(idx)),pcd_path=pcd_path,som_path=som_path,changed=self.changed)
            save_intermediate_element_for_SGG(obs, pcd, idx, pose_curr,
                                              camera_param_to_camera_matrix(self.env_camera_matrix), path=save_path)
    def updateSceneGraph(self,save_path,idx,exp_map):
        som_path = os.path.join(save_path,"som/")
        if not os.path.exists(som_path):
            os.makedirs(som_path)
        self.objects,self.bg_objects,self.changed = removalOverlap(self.cfg_detection,self.objects,self.bg_objects,self.changed)
        # draw_voxel_objects(self.objects,self.bg_objects,idx,save_path)
        pcd_path = os.path.join(save_path,"pcd/")
        if not os.path.exists(pcd_path):
            os.makedirs(pcd_path)
        draw_voxel_objects(self.objects,self.bg_objects,idx,pcd_path)
        som_pth_step = som_path + str(idx) +"/"
        if not os.path.exists(som_pth_step):
            os.makedirs(som_pth_step)
        relation_list,image_list,image_objects_dict,image_pair_dict,check_relation=prepareRelationForCOG(self.cfg_detection,self.objects,self.changed,som_pth_step,idx)
        ######## get surrounding objects#####
        surrounding_objects={}
        for relation in relation_list :
            if relation[0] in surrounding_objects.keys() :
                surrounding_objects[relation[0]].append(relation[1])
                surrounding_objects[relation[0]] = list(set(surrounding_objects[relation[0]]))
            else :
                surrounding_objects[relation[0]] = [relation[1]]
            if relation[1] in surrounding_objects.keys() :
                surrounding_objects[relation[1]].append(relation[0])
                surrounding_objects[relation[1]] = list(set(surrounding_objects[relation[1]]))
            else :
                surrounding_objects[relation[1]] = [relation[0]]
        generateSoMImage(self.cfg_detection,image_objects_dict,self.objects,self.reses_history)
        for key,value in image_pair_dict.items() :
            relation_local=COG_one(key,value,self.cog_tokenizer,self.cog_model)
            for obj1,dict in relation_local.items() :
                if obj1 not in self.scene_graph.keys() :
                    dict1={}
                    for obj2,relation in dict.items() :
                        dict1[obj2]=[relation]
                    self.scene_graph[obj1] = dict1
                else :
                    for obj2,relation in dict.items() :
                        if obj2 in self.scene_graph[obj1].keys() :
                            self.scene_graph[obj1][obj2].append(relation)
                        else :
                            self.scene_graph[obj1][obj2]=[relation]
        for key,value in self.scene_graph.items() :
            print(key,value)
        self.changed=[]
        obj_index=1
        voxels_detection = torch.zeros(
            self.num_scene,self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,
            self.max_height - self.min_height
            ).float().to(self.device)
        wall=None
        ####obj_index =1 indicates wall
        for name,bg in self.bg_objects.items() :
            if (name == "wall" or name == "wall-wood") and bg is not None :
                indices_3=np.array(np.unravel_index(bg['voxel_index'], (self.args.map_size_cm//self.resolution,self.args.map_size_cm//self.resolution,self.max_height-self.min_height),order='C')).T # n*3
                indices_3 = torch.from_numpy(indices_3).to(self.device)
                index =(indices_3 >torch.tensor([0,0,0]).to(self.device)) & (indices_3 < torch.tensor([self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,self.max_height - self.min_height]).to(self.device))
                index_new = torch.where(index[:,0]&index[:,1]&index[:,2])[0]
                voxels_detection[0,indices_3[index_new,1],indices_3[index_new,0],indices_3[index_new,2]] = obj_index  
                wall = indices_3[index_new,:2].cpu().numpy()
        obj_index +=1
        for obj in self.objects :
            indices_3=np.array(np.unravel_index(obj['voxel_index'],(self.args.map_size_cm//self.resolution,self.args.map_size_cm//self.resolution,self.max_height-self.min_height),order='C')).T # n*3
            indices_3 = torch.from_numpy(indices_3).to(self.device)
            index =(indices_3 >torch.tensor([0,0,0]).to(self.device)) & (indices_3 < torch.tensor([self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,self.max_height - self.min_height]).to(self.device))
            index_new = torch.where(index[:,0]&index[:,1]&index[:,2])[0]
            voxels_detection[0,indices_3[index_new,1],indices_3[index_new,0],indices_3[index_new,2]] = obj_index  
            obj_index +=1
        max_h, min_h = self.max_height, self.min_height
        min_z = int(25 / self.z_resolution - min_h)
        mid_z = int(self.agent_height / self.z_resolution - min_h)
        max_z = int((self.agent_height + 50) / self.z_resolution - min_h)
        agent_height_proj_detection = voxels_detection[..., min_z:max_z].max(3)[0]
        obstacle_map = agent_height_proj_detection[0]/ self.map_pred_threshold
        outlier_region = torch.where(exp_map==0)
        obstacle_map[outlier_region[0],outlier_region[1]]=-1
        return self.objects,self.bg_objects,surrounding_objects,obstacle_map,wall
    def getGlobalMap(self,map_before,ori_obs,eve_angle,current_poses,idx,save_path):
        points_pose = current_poses.clone()
        points_pose[:, 2] = points_pose[:, 2] * np.pi / 180
        points_pose[:, :2] = points_pose[:, :2] * 100
        ori_obs = torch.from_numpy(ori_obs.transpose(1, 2, 0)).to(self.device).float()
        ori_depth = preprocess_depth(ori_obs[:, :, 3].unsqueeze(2), 0.5, 5.0).unsqueeze(0)
        ori_point_cloud_t = du.get_point_cloud_from_z_t(ori_depth, self.env_camera_matrix, self.device, scale=self.du_scale)
        ori_agent_view_t, trans_matrix = du.transform_camera_view_t(ori_point_cloud_t, self.agent_height, eve_angle, self.device, need_transmatrix=True)
        # agent_view_t, points_pose_matrix = du.transform_pose_t(ori_agent_view_t, self.shift_loc, self.device, need_transmatrix=True)
        world_view_t,points_pose_matrix=du.transform_pose_t(ori_agent_view_t,points_pose[0].cpu().numpy(), self.device, need_transmatrix=True)
        voxels_detection = torch.zeros(
            self.num_scene,self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,
            self.max_height - self.min_height
            ).float().to(self.device)
        self.forward_detection(ori_obs.permute(2, 0, 1).unsqueeze(0), world_view_t.squeeze(), idx, points_pose_matrix @ trans_matrix ,save_path)
        obj_index=1
        wall=[]
        ####obj_index =1 indicates wall
        for name,bg in self.bg_objects.items() :
            if (name == "wall" or name == "wall-wood") and bg is not None :
                indices_3=np.array(np.unravel_index(bg['voxel_index'], (self.args.map_size_cm//self.resolution,self.args.map_size_cm//self.resolution,self.max_height-self.min_height),order='C')).T # n*3
                indices_3 = torch.from_numpy(indices_3).to(self.device)
                index =(indices_3 >torch.tensor([0,0,0]).to(self.device)) & (indices_3 < torch.tensor([self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,self.max_height - self.min_height]).to(self.device))
                index_new = torch.where(index[:,0]&index[:,1]&index[:,2])[0]
                voxels_detection[0,indices_3[index_new,1],indices_3[index_new,0],indices_3[index_new,2]] = obj_index  
                wall.append(obj_index)
        obj_index +=1
        for obj in self.objects :
            indices_3=np.array(np.unravel_index(obj['voxel_index'],(self.args.map_size_cm//self.resolution,self.args.map_size_cm//self.resolution,self.max_height-self.min_height),order='C')).T # n*3
            indices_3 = torch.from_numpy(indices_3).to(self.device)
            index =(indices_3 >torch.tensor([0,0,0]).to(self.device)) & (indices_3 < torch.tensor([self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,self.max_height - self.min_height]).to(self.device))
            index_new = torch.where(index[:,0]&index[:,1]&index[:,2])[0]
            voxels_detection[0,indices_3[index_new,1],indices_3[index_new,0],indices_3[index_new,2]] = obj_index  
            obj_index +=1
        max_h, min_h = self.max_height, self.min_height
        XYZ_cm_std = world_view_t.float()
        XYZ_cm_std[..., :2] = torch.floor(XYZ_cm_std[..., :2] / self.resolution)
        XYZ_cm_std[..., 2] = torch.floor((XYZ_cm_std[..., 2] / self.z_resolution) - min_h) 
        XYZ_cm_std = XYZ_cm_std.int().reshape(-1,3)
        index =(XYZ_cm_std >torch.tensor([0,0,0]).to(self.device)) & (XYZ_cm_std < torch.tensor([self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,
            self.max_height - self.min_height]).to(self.device))
        index_new = torch.where(index[:,0]&index[:,1]&index[:,2])[0]
        XYZ_cm_std = XYZ_cm_std[index_new]
        voxels = torch.zeros(
            self.num_scene,self.map_size_cm // self.resolution, self.map_size_cm // self.resolution,
            self.max_height - self.min_height
        ).float().to(self.device)
        
        
        voxels[0,XYZ_cm_std[:,1],XYZ_cm_std[:,0],XYZ_cm_std[:,2]]=1.

        min_z = int(25 / self.z_resolution - min_h)
        mid_z = int(self.agent_height / self.z_resolution - min_h)
        max_z = int((self.agent_height + 50) / self.z_resolution - min_h)
        agent_height_proj = voxels[..., min_z:max_z].sum(3)
        agent_height_proj_detection = voxels_detection[..., min_z:max_z].max(3)[0]
        agent_height_stair_proj = voxels[..., mid_z-5:mid_z].sum(3)
        all_height_proj = voxels.sum(3)
        current_map = torch.zeros(map_before.shape[0],map_before.shape[1],
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution,
                                 ).to(self.device)
        
        current_map[0,0] =torch.clamp(agent_height_proj/ self.map_pred_threshold, min = 0.0, max = 1.0)[0]
        current_map[0,1] = torch.clamp(all_height_proj / self.exp_pred_threshold, min = 0.0, max = 1.0)[0]
        current_map[0,4] = agent_height_proj_detection[0]/ self.map_pred_threshold

        diff_ob_ex = current_map[:, 1:2, :, :] - self.max_pool(current_map[:, 0:1, :, :]) # 整个Voxel高度的地图 - Agent视野投影
        diff_ob_ex[diff_ob_ex > 0.8] = 1.0
        diff_ob_ex[diff_ob_ex != 1.0]  = 0.0
       
        map2 = torch.cat((map_before.unsqueeze(1), current_map.unsqueeze(1)), 1)
        map_pred, _ = torch.max(map2, 1)
        outlier_region = torch.where(map_pred[0,1]==0)
        map_pred[0,4,outlier_region[0],outlier_region[1]]=-1
        for i in range(eve_angle.shape[0]):
            if eve_angle[i] == 0:
                map_pred[i, 0:1, :, :][diff_ob_ex[i] == 1.0] = 0.0
        current_map_stair = current_map.clone().detach()
        current_map_stair[0,0]=torch.clamp(agent_height_stair_proj[:, :, :] / self.map_pred_threshold, min = 0.0, max = 1.0)
        stair_mask = torch.zeros(self.map_size_cm // self.resolution, self.map_size_cm // self.resolution).to(self.device)
        s_y = int(current_poses[0][1]*100/5)
        s_x = int(current_poses[0][0]*100/5)
        limit_up = self.map_size_cm // self.resolution - self.stair_mask_radius - 1
        if s_y > limit_up:
            s_y = limit_up
        if s_y < self.stair_mask_radius:
            s_y = self.stair_mask_radius
        if s_x > limit_up:
            s_x = limit_up
        if s_x < self.stair_mask_radius:
            s_x = self.stair_mask_radius
        stair_mask[int(s_y-self.stair_mask_radius):int(s_y+self.stair_mask_radius), int(s_x-self.stair_mask_radius):int(s_x+self.stair_mask_radius)] = self.stair_mask
        current_map_stair[0, 0:1, :] *= stair_mask
        current_map_stair[0, 1:2, :] *= stair_mask
        diff_ob_ex = current_map_stair[:, 1:2, :, :] - current_map_stair[:, 0:1, :, :]
        diff_ob_ex[diff_ob_ex>0.8] = 1.0
        diff_ob_ex[diff_ob_ex!=1.0] = 0.0
        maps3 = torch.cat((map_before.unsqueeze(1), current_map_stair.unsqueeze(1)), 1)
        map_pred_stair, _ = torch.max(maps3, 1)
        for i in range(eve_angle.shape[0]):
            if eve_angle[i] == 0:
                map_pred_stair[i, 0:1, :, :][diff_ob_ex[i] == 1.0] = 0.0
        return map_pred,map_pred_stair
    def forward(self, obs, ori_obs, pose_obs, maps_last, poses_last, eve_angle, origins, idx,save_path):
        current_poses = get_new_pose_from_rel_pose_batch(poses_last, pose_obs)
       
        maps_ours,map_stair_ours = self.getGlobalMap(maps_last,ori_obs,eve_angle,current_poses,idx,save_path)
       
        return maps_ours, map_stair_ours, current_poses
    def reset(self):
        self.global_classes = set()
        self.objects = MapObjectList(device=self.device)
        self.bg_objects = {c: None for c in self.BG_CLASSES}
        self.changed=[]
        self.selected_view=[]
    def find_Maximum_View(self,obj_id,start_idx,end_idx):
        maximum_len = 0
        select_idx = -1
        # import pdb
        # pdb.set_trace()
        for i in range(start_idx,end_idx+1) :
            if i in self.reses_history.keys() :
                res = self.reses_history[i]
                cam_K,pose = res[2][:3,:3],res[3].cpu().numpy()
                cam_intrinsic = cam_K[:3, :3]
                mask = mapObjects(cam_intrinsic,np.linalg.inv(pose),self.args.env_frame_height,self.args.env_frame_width,self.objects)
                index = np.where(mask == obj_id)[0]
                if len(index) >= maximum_len :
                    select_idx = i 
                    maximum_len = len(index)
        if select_idx != -1 :
            self.selected_view.append(select_idx)
        if select_idx == -1 :
            print("no satisfied")
            select_idx = start_idx
        return select_idx
    def get_mask(self, mask_range):
        size = int(mask_range) * 2
        mask = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if ((i + 0.5) - (size // 2)) ** 2 + ((j + 0.5) - (size // 2)) ** 2 <= mask_range ** 2:
                    mask[i, j] = 1
        return mask