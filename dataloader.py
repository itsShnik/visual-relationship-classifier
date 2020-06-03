import torch.utils.data as Data
import json
from PIL import image


class DatasetLoader(Data.Dataset):

    def __init__(self, config):
        self.config = config
        # load all the files here

        if config.RUN_MODE == 'train':
            # load the train relationships
            self.rel = json.load(open('data/relationships_train.json', 'r'))

        # define the data size
        self.data_size = len(self.rel)

    def load_regions(img, obj):
        # extract the region from the image
        cropped_region = img.crop((obj['x'], obj['y'], obj['w'], obj['h']))
        return cropped_region

    def load_image(path):
        img = Image.open(path)
        img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        # custom get item method goes here

        rel = self.rel[idx]

        # load the image
        img = load_image(self.image_path + rel['image_id'] + '.jpg')

        # load the subject and objects
        subj = load_regions(img, rel['subject'])
        obj = load_regions(img, rel['object'])

        # the predicate gives the class
        cls = rel['predicate']

        return img, subj, obj, cls

    def __len__(self):
        # custom length function goes here
        return self.data_size
