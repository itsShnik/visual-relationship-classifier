# first of all create a dataloader to pass all the objects
import _init_paths
import torch
import torch.utils.data as Data
import json
from PIL import Image
import numpy as np
import glob
import random
import time
from torchvision import transforms
import pickle
import os
import torch.nn as nn
import base64
from itertools import islice, cycle
from vis_rel.function.config import config, update_config
from vis_rel.modules.frcnn_classifier import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

update_config('cfgs/vis_rel/frcnn.yaml')

class COCOLoader(Data.Dataset):

    def __init__(self):

        self.obj_path_list = glob.glob('data/coco/vgbua_res101_precomputed/*faster_rcnn_genome/*.json')

        print("Total images are: ", len(self.obj_path_list))

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        # that's all we want
        self.data_size = len(self.obj_path_list)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        path = self.obj_path_list[idx]
        image_id = int(path.split('/')[-1].replace('.json', ''))
        objects = json.load(open(path))
        im_info = torch.tensor((objects['image_w'], objects['image_h']))

        # load the objects
        boxes = np.frombuffer(base64.decodebytes(objects['boxes'].encode()), dtype=np.float32).reshape((objects['num_boxes'], -1))

        # sample 20 objects
        boxes = list(boxes)

        if len(boxes) > 20:
            boxes = random.sample(list(boxes), 20)
        else:
            boxes = list(islice(cycle(list(boxes)), 20))

        # convert np arrays to tensor
        boxes = [torch.from_numpy(box) for box in boxes]

        # stack to tensor
        boxes = torch.stack(boxes)

        # now load the image
        img = Image.open(glob.glob('data/coco/*/*' + str(image_id) + '.jpg')[0])
        img = img.convert('RGB')
        img = self.transform(img)

        return img, boxes, im_info, torch.tensor(image_id)

# give a batch size
batch_size = 2048

# now create a dataset
coco_dataset = COCOLoader()
print('created a dataset loader')

data_size = coco_dataset.data_size

coco_dataloader = Data.DataLoader(
            coco_dataset,
            batch_size,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
            drop_last = True
        )

print("Built the dataloader....")

# Load the net
model = Net(config)
model.cuda()
model.eval()
model = nn.DataParallel(model, device_ids = [0,1,2,3])

print('Loaded the net')

# Load the state dict
state_path = 'output/output/vis_rel/ckpt_frcnn_train+val_low_lr_epoch7.pkl'
state_dict = torch.load(state_path)['state_dict']
model.load_state_dict(state_dict)

print('Loaded the checkpoint')

# define a threshold
threshold = 0.75

# dic
dic = {}

# now the evaluator
with torch.no_grad():
    for step, (img, boxes, im_info, image_id) in enumerate(coco_dataloader):
        image_id = image_id.detach().cpu()
        # now for every pair of objects in each image

        # create an empty list for each of the image
        for k in range(len(image_id)):
            if str(int(image_id[k])) not in dic:
                dic[str(int(image_id[k]))] = []

        rel_count = 0
        start_time = time.time()
        for i in range(len(boxes[0])):
            for j in range(len(boxes[0])):
                if i != j:
                    inputs = {
                                'subj_bbox':boxes[:,i],
                                'obj_bbox':boxes[:,j],
                                'union_bbox':torch.stack([torch.min(boxes[:,i][:,0], boxes[:,j][:,0]), torch.min(boxes[:,i][:,1], boxes[:,j][:,1]), torch.max(boxes[:,i][:,2], boxes[:,j][:,2]), torch.max(boxes[:,i][:,3], boxes[:,j][:,3])], -1),
                                'im_info': im_info,
                                'image': img
                            }

                    feats, pred = model(inputs)

                    # take softmax over the predictions
                    pred = pred.softmax(-1)

                    # keep only those in which the max prob is greater than
                    # threshold and the class is not 21 (index 20)
                    keep = (torch.max(pred, -1).values > threshold) & (torch.argmax(pred,-1) < 20)


                    # things to save
                    save_image_id = image_id[keep]

                    # Keep a rel_count
                    rel_count += save_image_id.__len__()

                    save_feats = feats.detach().cpu()[keep]
                    save_pred = pred.detach().cpu()[keep]
                    save_subj_bbox = inputs['subj_bbox'][keep].clone().detach().cpu()
                    save_obj_bbox = inputs['obj_bbox'][keep].clone().detach().cpu()

                    # now append relevant bounding boxes of relevant images to their corresponding dictionary
                    for k in range(len(save_image_id)):
                        dic[str(int(save_image_id[k]))].append({'feats':save_feats[k], 'subj_bbox':save_subj_bbox[k], 'obj_bbox':save_obj_bbox[k], 'pred':save_pred[k]})

        end_time = time.time()

        print("Step {}/{}, Time taken is : {}, Avg rel count is {}".format(step, int(data_size/batch_size), end_time-start_time, rel_count/batch_size))
                    

# pickle the dic
print('saving the relationships')
for k,v in dic.items():
    torch.save(v, open(f'data/vqa_relationships/{str(k)}.pkl', 'wb'))
print('Done!')

