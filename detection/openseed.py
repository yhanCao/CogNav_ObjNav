import os
import sys

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)
from tqdm import trange
import torch
from torchvision import transforms
from tqdm import trange
from utils.arguments import load_opt_command
import pickle as pkl
import gzip
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer

def InitDetection():
    global model,transform
    opt = load_opt_command()
    pretrained_pth = "model/pretrained_models/openseed_swinl_pano_sota.pt"
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    thing_classes=[]
    with open('configs/panoptic_categories_nomerge.txt', 'r') as file:
        for line in file:
                thing_classes.append(line.strip())
    # thing_classes = ['car','person','traffic light', 'truck', 'motorcycle']
    stuff_classes = []
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
    
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
    return thing_classes,thing_colors
def convertMask(mask_origin,ids):
    label_length=len(ids)
    masks=np.zeros((label_length,mask_origin.shape[0],mask_origin.shape[1]),dtype=np.bool_)
    for i in range(label_length) :
        masks[i]=(mask_origin == ids[i])
    return masks
def detection(image_ori,idx,visual_save_path=None,detection_save_path=None):
    with torch.no_grad():
        width = image_ori.size[0]
        height = image_ori.size[1]
        image = transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
        batch_inputs = [{'image': images, 'height': height, 'width': width}]
        outputs = model.forward(batch_inputs)
        visual = Visualizer(image_ori, metadata=model.model.metadata)
        pano_seg = outputs[-1]['panoptic_seg'][0]
        pano_seg_info = outputs[-1]['panoptic_seg'][1]
        for i in range(len(pano_seg_info)):
            if pano_seg_info[i]['category_id'] in model.model.metadata.thing_dataset_id_to_contiguous_id.keys():
                pano_seg_info[i]['category_id'] = model.model.metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                pano_seg_info[i]['category_name'] = model.model.metadata.thing_classes[pano_seg_info[i]['category_id']]
            else:
                pano_seg_info[i]['isthing'] = False
                pano_seg_info[i]['category_id'] = model.model.metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
                pano_seg_info[i]['category_name'] = model.model.metadata.thing_classes[pano_seg_info[i]['category_id']]
        ###visualization
        demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info) # rgb Image
        demo.save(os.path.join(visual_save_path, str(idx)+'.png'))
        ### save result
        id = [pano['id']for pano in pano_seg_info]
        mask = convertMask(pano_seg.cpu().numpy(),id)
        labels = [model.model.metadata.thing_classes[pano['category_id']] for pano in pano_seg_info]
        if len(labels) > 0 :
            visual_feature=torch.stack([pano['semantic_feature'] for pano in pano_seg_info])
            bbox = torch.stack([pano['bbox'] for pano in pano_seg_info])
            category_id = [pano['category_id']for pano in pano_seg_info]
            category_name = [pano['category_name']for pano in pano_seg_info]
            visual_feature=torch.stack([pano['semantic_feature'] for pano in pano_seg_info])
            text_feature = torch.stack([getattr(model.model.sem_seg_head.predictor.lang_encoder,'default_text_embeddings')[pano['category_id']] for pano in pano_seg_info])
            if len(category_id) > 0 :
                results = {
                    "xyxy": bbox.cpu().numpy(),
                    "class_id": np.asarray(category_id),
                    "mask": mask,
                    "classes": category_name,
                    "image_feats": visual_feature.cpu().numpy(),
                    "text_feats": text_feature.cpu().numpy(),
                }
                if detection_save_path is not None:
                    with gzip.open(detection_save_path+str(idx)+".pkl.gz", "wb") as f:
                        pkl.dump(results, f)
        else :
            results=None
    return results
def read_pkl(path_file):
    output = open(path_file, 'rb')
    res = pkl.load(output)
    return res
def main(dir,start,end):
    for i in trange(start,end+1):
        img_name = os.path.join(dir,str(i)+".jpg")
        detection_path = os.path.join(dir,"detection/")
        visualization_path = os.path.join(dir,"visual/")
        if not os.path.exists(detection_path):
            os.makedirs(detection_path)
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path)
        detection(img_name,i,visualization_path,detection_path)
        # save_dict_name = os.path.join(dir,"save_dict_"+str(i)+".pkl")
        # res = read_pkl(save_dict_name)
        # print(res.keys())
        # pcd_array = res['pcd_frame']['points']
if __name__ == "__main__":
    InitDetection()
    start,end = -1,171
    main(dir,start,end)
