# First create a dataloader
import torch
import torch.utils.data as Data
import json
from PIL import Image
import numpy as np
import glob
import time
import transforms as T
from torchvision import transforms
import utils
import pickle

#def my_collate(batch):
#    img = [item[1] for item in batch]
#    img_id = [item[0] for item in batch]
#    img_id = torch.LongTensor(img_id)
#    return [img_id, img]


class COCOLoader(Data.Dataset):

    def __init__(self):

        # Load all the images from coco
        self.images = glob.glob('../data/coco/*/*.jpg')

        # self data size
        self.data_size = len(self.images)

        # define the required transform on images
        self.resize = T.RandomResize([800], max_size=1331)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):

        img_path = self.images[idx]
        img_id = torch.tensor(int(img_path.split('/')[-1].replace('.jpg', '').split('/')[-1].split('_')[-1]))
        img = Image.open(img_path)
        img = img.convert('RGB')
        img, _ = self.resize(img)
        img = self.transform(img)

        #print("The shape of image is : ", img.shape)

        return img, img_id

    def __len__(self):
        return self.data_size

# batch
batch_size = 1

# create an object for dataset
cocodataset = COCOLoader()

print("Built the dataset....")

# data size
data_size = cocodataset.data_size

# create a dataloader
coco_dataloader = Data.DataLoader(
            cocodataset,
            batch_size,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
            drop_last = True
            #collate_fn = utils.collate_fn
        )

print("Built the dataloader....")

# define a model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.cuda()
model.eval()

# parallelize the data
model = torch.nn.DataParallel(model, device_ids = [0,1,2,3])

print("loaded and parallelized the model")

# image_wise_boxes
image_boxes = {}

# now eval for each batch

start_time = time.time()

avg = 0

for step, (img, img_id) in enumerate(coco_dataloader):

    img_id = img_id.cuda()
    img = img.cuda()
    #device = torch.device('cuda')
    #img = img.to(device)
    #print("in evaluator img device is : ", img.device)

    outputs = model(img)

    # keep only predictions with 0.95+ confidence
    pred_boxes = outputs['pred_boxes'].detach().cpu()
    probas = outputs['pred_logits'].softmax(-1)[:, :, :-1].detach().cpu()
    keep = probas.max(-1).values > 0.92

    new = 0

    for i in range(len(pred_boxes)):
        image_boxes[str(int(img_id[i]))] = {'im_info':list(img[i].shape) , 'obj_list': list(pred_boxes[i][keep[i]])}
        new += len(pred_boxes[i][keep[i]])

    avg = avg + (new - avg)/(step+1)

    print("\rstep {}/{}: Average {} boxes".format(step, data_size/batch_size, int(avg)), end='  ')

end_time = time.time()

print("Total time taken is : ", end_time-start_time)


# save to pkl file
print("saving to pickle file...")
start_time = time.time()
pickle.dump(image_boxes, open('../data/coco_objects.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
end_time = time.time()
print("Total time taken to picle is : ", end_time-start_time)
