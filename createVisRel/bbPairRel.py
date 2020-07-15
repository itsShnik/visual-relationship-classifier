import _init_paths
import numpy as np
import json
import base64
import copy
import glob
import time
import random
import matplotlib.pyplot as plt
import pickle

import torch.nn as nn
import torch
import torch.utils.data as Data
from PIL import Image
from torchvision import transforms

from vis_rel.function.config import config, update_config
from vis_rel.modules.frcnn_classifier import Net

update_config('cfgs/vis_rel/frcnn.yaml')

"""
Load the bounding box pairs
"""
print('Loading the bb pairs dataset')
bb_pairs_dataset = pickle.load(open('data/bb_pairs_dataset.pkl', 'rb'))

print('Loaded the bb pairs dataset')

"""
The dataloader for the dataset
"""

import random

class DatasetLoader(Data.Dataset):
    
    def __init__(self, path, bb_pairs_dataset):
        
        self.image_dir = path
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.bb_pairs = bb_pairs_dataset
        self.data_size = len(self.bb_pairs)
        
    def __getitem__(self, idx):
        
        bb_pair = self.bb_pairs[idx]
        
        img = Image.open(glob.glob(self.image_dir + '/*/*' + str(bb_pair['image_id']) + '.jpg')[0])
        img = img.convert('RGB')
        img = self.transform(img)
        
        inputs = {}
        
        # convert inputs to tensors
        inputs['subj_bbox'] = torch.from_numpy(bb_pair['subj_bbox'])
        inputs['obj_bbox'] = torch.from_numpy(bb_pair['obj_bbox'])
        inputs['union_bbox'] = torch.from_numpy(bb_pair['union_bbox'])
        
        inputs['im_info'] = torch.tensor(bb_pair['im_info'])
        inputs['image'] = img
        
        # image_id
        image_ids = torch.tensor(bb_pair['image_id'])
        
        return inputs, image_ids
        
        
    def __len__(self):
        return self.data_size

"""
The evaluator
"""
# batch_size
batch_size = 16

# Load the dataset
val_dataset = DatasetLoader('data/coco', bb_pairs_dataset)

val_dataloader = Data.DataLoader(
                    val_dataset,
                    batch_size,
                    shuffle = False,
                    num_workers = 4,
                    pin_memory = True,
                    sampler = None,
                    drop_last = True
                )

print('Loaded the dataset')

# os env
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# Load the net
model = Net(config)

# define a softmax obj
soft = nn.Softmax(-1)

# Load the state dict
path = 'output/output/vis_rel/ckpt_frcnn_train+val_low_lr_epoch7.pkl'
state_dict = torch.load(path, map_location=torch.device('cpu'))['state_dict']
new_state_dict = {k.replace('module.', ''):state_dict[k] for k in state_dict}
model.load_state_dict(new_state_dict)

model.eval()
model = nn.DataParallel(model, device_ids = [0,1,2,3]).cuda()

# Load the idx to label relationship
rel_classes = json.load(open('data/relationship_classes.json'))
class_rel = {v:k for k, v in rel_classes.items()}
print(class_rel)

# define a threshold
threshold = 0.6

relationships = {}

start_time = time.time()

with torch.no_grad():
    for step, (
            inputs,
            image_ids
        ) in enumerate(val_dataloader):

        for k, v in inputs.items():
            inputs[k] = v.to(torch.device('cuda'))
        feats, pred = model(inputs)

        # softmax over pred
        pred = soft(pred)

        for i in range(len(pred)):
            pred_ind = int(torch.argmax(pred[i]))
            pred_val = torch.max(pred[i])

            if pred_ind < 20 and pred_val > threshold:
                temp_rel = {
                    'predicate': str(class_rel[int(pred_ind)]),
                    'features': feats.cpu().detach().tolist(),
                    'subj_bbox': inputs['subj_bbox'][i].cpu().detach().tolist(),
                    'obj_bbox': inputs['obj_bbox'][i].cpu().detach().tolist()
                }

                if str(int(image_ids[i])) not in relationships:
                    relationships[str(int(image_ids[i]))] = []

                relationships[str(int(image_ids[i]))].append(temp_rel)

        print("\rProgress {}/{}".format(step, val_dataset.data_size/batch_size), end='  ')


for k, v in relationships.items():
    print("\rImage id : {}".format(str(k)), end=' ')
    pickle.dump(v, open('data/coco/vqa_relationships/' + str(k) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

end_time = time.time()

print('Time taken is : ', end_time - start_time)

