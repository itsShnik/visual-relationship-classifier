import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class Net(nn.Module):
    def __init__(self, config, num_classes = 21):
        super(Net, self).__init__()
        
        self.backbone_network = models.resnet101(pretrained=True)
        
        # set the parmas in backbone non trainable
        for params in self.backbone_network.parameters():
            params.requires_grad = False
            
        
        # projection layers to project into smaller dimension
        self.subj_proj = nn.Linear(1000, config.PROJ_DIM)
        self.obj_proj = nn.Linear(1000, config.PROJ_DIM)
        self.union_proj = nn.Linear(1000, config.PROJ_DIM)

        # projection from PROJ_DIM + POS_DIM to EMBED_DIM
        self.subj_emb = nn.Linear(config.PROJ_DIM, config.EMBED_DIM)
        self.obj_emb = nn.Linear(config.PROJ_DIM, config.EMBED_DIM)
        self.union_emb = nn.Linear(config.PROJ_DIM, config.EMBED_DIM)
        
        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(3 * config.PROJ_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, num_classes))
        
    def forward(self, inputs):
        
        subj = inputs['subject']
        obj = inputs['object']
        union = inputs['union']
        
        subj = self.backbone_network(subj)
        obj = self.backbone_network(obj)
        union = self.backbone_network(union)

        subj_proj = self.subj_proj(subj)
        obj_proj = self.obj_proj(obj)
        union_proj = self.union_proj(union)
        
        embedding = torch.cat((subj_proj, obj_proj, union_proj), -1)
        
        pred = self.classifier(embedding)
        
        return pred
