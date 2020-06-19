import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

# to find the visual features
from vis_rel.utils.frcnn_embed import FastRCNN

class Net(nn.Module):
    def __init__(self, config, num_classes = 21):
        super(Net, self).__init__()

        self.frcn_adapter = FastRCNN(config, average_pool=True, final_dim = config.FINAL_DIM) 

        self.embed_proj = nn.Linear(3 * config.FINAL_DIM, config.FINAL_DIM)
        
        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(config.FINAL_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, num_classes))
        
    def forward(self, inputs):
        
        feat_dict = self.frcn_adapter(inputs)

        # obtain the subj, obj and union features
        subj_feats = feat_dict['subj']
        obj_feats = feat_dict['obj']
        union_feats = feat_dict['union']

        # concatenate the object features
        feats = torch.cat((subj_feats, obj_feats, union_feats), -1)

        # project the features to the embedding dimension
        embedding = self.embed_proj(feats)
        
        # classify into the classes
        pred = self.classifier(embedding)
        
        return pred
