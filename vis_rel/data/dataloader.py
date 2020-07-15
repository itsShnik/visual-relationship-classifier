import torch
import torch.utils.data as Data
import json
from PIL import Image
import numpy as np
from torchvision import transforms

class DatasetLoader(Data.Dataset):

    def __init__(self, config):
        self.config = config
        # load all the files here

        if 'train' in config.RUN_MODE.split('+'):
            # load the train relationships
            self.rel = json.load(open(config.TRAIN_PATH))
            print("Loaded the training relationships from json")

        if 'val' in config.RUN_MODE.split('+'):
            if self.rel.__len__() > 0:
                self.rel.extend(json.load(open(config.VAL_PATH)))
            else:
                self.rel = json.load(open(config.VAL_PATH))
            print("Loaded the validation relationships from json")

        if config.RUN_MODE == 'test':
            self.rel = json.load(open(config.TEST_PATH))
            print("Loaded the test relationships from json")


        # toy dataset
        # self.rel = self.rel[:1024]

        self.cls_to_ix = json.load(open(config.REL_PATH))

        self.num_rel_classes = self.cls_to_ix.__len__()
        # define the data size
        self.data_size = len(self.rel)

        #if config.USE_FRCNN_FEATURES:
        #    self.transform = transforms.ToTensor()

        #else:
        # transforms and normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])


    def load_regions(self, img, obj):
        # extract the region from the image
        if obj['h'] <= 1:
            obj['h'] = 1
        if obj['w'] <= 1:
            obj['w'] = 1
        # extract the regions from the image
        x1, y1, w, h = obj['x'], obj['y'], obj['w'], obj['h']
        x2, y2 = x1 + w, y1 + h

        # bbox
        bbox = torch.tensor([x1, y1, x2, y2])

        if self.config.USE_FRCNN_FEATURES:
            return None, bbox

        # crop the region
        cropped_region = img.crop((x1, y1, x2, y2))
        return cropped_region, bbox

    def load_regions_union(self, img, obj1, obj2):
        # find the union of the regions
        x1 = min(obj1['x'], obj2['x'])
        y1 = min(obj1['y'], obj2['y'])

        x2 = max(obj1['x']+obj1['w'], obj2['x']+obj2['w'])
        y2 = max(obj1['y']+obj1['h'], obj2['y']+obj2['h'])

        w, h = x2 - x1, y2 - y1

        #bbox
        bbox = torch.tensor([x1, y1, x2, y2])

        if self.config.USE_FRCNN_FEATURES:
            return None, bbox

        return img.crop((x1, y1, x2, y2)), bbox

    def load_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')

        im_info = torch.tensor(img.size)

        return img, im_info

    def __getitem__(self, idx):
        # custom get item method goes here

        rel = self.rel[idx]

        # load the image
        img, im_info = self.load_image(self.config.IMAGES_PATH + str(rel['image_id']) + '.jpg')

        # load the subject and objects
        subj, subj_bbox = self.load_regions(img, rel['subject'])
        if subj is not None:
            subj = self.transform(subj)

        obj, obj_bbox = self.load_regions(img, rel['object'])
        if subj is not None:
            obj = self.transform(obj)

        # the union of the regions
        union, union_bbox = self.load_regions_union(img, rel['subject'], rel['object'])
        if subj is not None:
            union = self.transform(union)

        img = self.transform(img)

        # the predicate gives the class
        labels = self.cls_to_ix[rel['predicate']]

        # create a dict for inputs
        # check for which model are we using
        if not self.config.USE_FRCNN_FEATURES:
            inputs = {
                'subject': subj,
                'object': obj,
                'union': union,
                'subject_bbox': subj_bbox,
                'object_bbox': obj_bbox,
                'union_bbox': union_bbox
            }
        else:
            inputs = {
                'image': img,
                'subj_bbox': subj_bbox,
                'obj_bbox': obj_bbox,
                'union_bbox': union_bbox,
                'im_info': im_info
            }
        # convert class labels to torch tensors
        labels = torch.tensor(labels)

        return inputs, labels

    def __len__(self):
        # custom length function goes here
        return self.data_size
