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

        if config.RUN_MODE == 'train':
            # load the train relationships
            self.rel = json.load(open('data/relationships_train.json', 'r'))

        elif config.RUN_MODE == 'val':
            self.rel = json.load(open('data/relationships_val.json', 'r'))

        self.cls_to_ix = json.load(open('data/relationship_classes.json', 'r'))

        self.num_rel_classes = self.cls_to_ix.__len__()
        # define the data size
        self.data_size = len(self.rel)

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
        assert obj['w'] > 0 and obj['h'] > 0
        cropped_region = img.crop((obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']))
        return cropped_region

    def load_regions_union(self, img, obj1, obj2):
        # find the union of the regions
        x = min(obj1['x'], obj2['x'])
        y = min(obj1['y'], obj2['y'])

        x2 = max(obj1['x']+obj1['w'], obj2['x']+obj2['w'])
        y2 = max(obj1['y']+obj1['h'], obj2['y']+obj2['h'])

        return img.crop((x, y, x2, y2))

    def load_image(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        # custom get item method goes here

        rel = self.rel[idx]

        # load the image
        img = self.load_image('data/images/' + str(rel['image_id']) + '.jpg')

        #print("before cropping, subject, ({}, {})".format(subj['w'], subj['h']))
        #print("before cropping, object, ({}, {})".format(obj['w'], obj['h']))

        # load the subject and objects
        subj = self.load_regions(img, rel['subject'])
        subj = self.transform(subj)
        #print("after cropping, subject, {}".format(subj.size))
        obj = self.load_regions(img, rel['object'])
        obj = self.transform(obj)
        #print("after cropping, object, {}".format(obj.size))

        # the union of the regions
        union = self.load_regions_union(img, rel['subject'], rel['object'])
        union = self.transform(union)

        img = self.transform(img)
        # the predicate gives the class
        labels = self.cls_to_ix[rel['predicate']]

        # create a dict for inputs
        inputs = {
            'subject': subj,
            'object': obj,
            'union': union,
            'img': img
        }
       
        # convert class labels to torch tensors
        labels = torch.tensor(labels)

        return inputs, labels

    def __len__(self):
        # custom length function goes here
        return self.data_size
