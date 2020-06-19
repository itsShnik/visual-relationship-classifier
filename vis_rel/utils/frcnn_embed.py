# torch imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# resnet related imports
from common.backbone.resnet.resnet import *
from common.backbone.resnet.resnet import Bottleneck, BasicBlock
from common.backbone.resnet.resnet import model_urls

# do some path settings to use roi_lib
import sys
import os
cwd = os.getcwd()
path = cwd + '/vis_rel/utils/frcnn/lib'
sys.path.append(path)

# roi related imports
# from model.roi_layers import ROIPool
# from model.roi_layers import ROIAlign
from detectron2.layers import ROIAlign

# utils
from common.utils.flatten import Flattener
from common.utils.bbox import coordinate_embeddings


class FastRCNN(nn.Module):
    def __init__(self, config, average_pool=True, final_dim=768):
        """
        :param config:
        :param average_pool: whether or not to average pool the representations
        :param final_dim:
        :param is_train:
        """
        super(FastRCNN, self).__init__()
        self.average_pool = average_pool
        self.final_dim = final_dim

        # about the resnet network
        self.stride_in_1x1 = config.NETWORK.IMAGE_STRIDE_IN_1x1
        self.c5_dilated = config.NETWORK.IMAGE_C5_DILATED
        self.num_layers = config.NETWORK.IMAGE_NUM_LAYERS
        self.pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.IMAGE_PRETRAINED,
                                                              config.NETWORK.IMAGE_PRETRAINED_EPOCH) if config.NETWORK.IMAGE_PRETRAINED != '' else None
        self.output_conv5 = config.NETWORK.OUTPUT_CONV5
        if self.num_layers == 18:
            self.backbone = resnet18(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                     expose_stages=[4])
            block = BasicBlock
        elif self.num_layers == 34:
            self.backbone = resnet34(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                     expose_stages=[4])
            block = BasicBlock
        elif self.num_layers == 50:
            self.backbone = resnet50(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                     expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
            block = Bottleneck
        elif self.num_layers == 101:
            self.backbone = resnet101(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                      expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
            block = Bottleneck
        elif self.num_layers == 152:
            self.backbone = resnet152(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                      expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
            block = Bottleneck
        else:
            raise NotImplemented

        # for roi align
        output_size = (14, 14)
        self.roi_align = ROIAlign(output_size=output_size, spatial_scale=1.0 / 16, sampling_ratio=2)

        # if object labels are available
        if config.NETWORK.IMAGE_SEMANTIC:
            self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
        else:
            self.object_embed = None
            self.mask_upsample = None

        # construct a head feature extractor
        self.roi_head_feature_extractor = self.backbone._make_layer(block=block, planes=512, blocks=3,
                                                                    stride=2 if not self.c5_dilated else 1,
                                                                    dilation=1 if not self.c5_dilated else 2,
                                                                    stride_in_1x1=self.stride_in_1x1)
        if average_pool:
            self.head = torch.nn.Sequential(
                self.roi_head_feature_extractor,
                nn.AvgPool2d(7 if not self.c5_dilated else 14, stride=1),
                Flattener()
            )
        else:
            self.head = self.roi_head_feature_extractor

        # if we need to freeze some layers
        if config.NETWORK.IMAGE_FROZEN_BN:
            for module in self.roi_head_feature_extractor.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = False

        frozen_stages = config.NETWORK.IMAGE_FROZEN_BACKBONE_STAGES
        if 5 in frozen_stages:
            for p in self.roi_head_feature_extractor.parameters():
                p.requires_grad = False
            frozen_stages = [stage for stage in frozen_stages if stage != 5]
        self.backbone.frozen_parameters(frozen_stages=frozen_stages,
                                        frozen_bn=config.NETWORK.IMAGE_FROZEN_BN)

        # downsample the object feats
        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * 2048 + (128 if config.NETWORK.IMAGE_SEMANTIC else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )

    def init_weight(self):
        if not self.image_feat_precomputed:
            if self.pretrained_model_path is None:
                pretrained_model = model_zoo.load_url(model_urls['resnet{}'.format(self.num_layers)])
            else:
                pretrained_model = torch.load(self.pretrained_model_path, map_location=lambda storage, loc: storage)
            roi_head_feat_dict = {k[len('layer4.'):]: v for k, v in pretrained_model.items() if k.startswith('layer4.')}
            self.roi_head_feature_extractor.load_state_dict(roi_head_feat_dict)
            if self.output_conv5:
                self.conv5.load_state_dict(roi_head_feat_dict)

    def bn_eval(self):
        if not self.image_feat_precomputed:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def obtain_feats(self, img_feats, bbox, indices, im_info):

        # firstly concatenate the indices with bbox to form rois
        rois = torch.cat((indices.unsqueeze(1), bbox), -1)

        # now we can obtain the roi_aligned features
        roi_align_res = self.roi_align(img_feats['body4'], rois).type(bbox.dtype)

        # find the features post roi align
        post_roialign = self.head(roi_align_res)

        # feats_to_downsample can only be post_roi_align
        # because there is no object labels
        feats_to_downsample = post_roialign

        # concatenate the coordinate embeddings
        coord_embed = coordinate_embeddings(
            torch.cat((bbox, im_info), -1),
            256
        )

        # concatenate the feats with coordinate embeddings
        feats_to_downsample = torch.cat((coord_embed.view((coord_embed.shape[0], -1)), feats_to_downsample), -1)

        # downsample to visual feature embedding dimension
        final_feats = self.obj_downsample(feats_to_downsample)

        return final_feats

    def forward(self, inputs):
        
        images = inputs['image']

        # calculate the image features from backbone
        # shape : (N, 1024, w, h)
        img_feats = self.backbone(images)

        subj_bbox = inputs['subj_bbox'].type(images.dtype)
        obj_bbox = inputs['obj_bbox'].type(images.dtype)
        union_bbox = inputs['union_bbox'].type(images.dtype)

        im_info = inputs['im_info'].type(images.dtype)

        # create an indices tensor
        indices = torch.tensor([*range(images.shape[0])]).type(images.dtype).cuda()

        subj_feats = self.obtain_feats(img_feats, subj_bbox, indices, im_info)
        obj_feats = self.obtain_feats(img_feats, obj_bbox, indices, im_info)
        union_feats = self.obtain_feats(img_feats, union_bbox, indices, im_info)

        feat_dict = {}
        feat_dict['subj'] = subj_feats
        feat_dict['obj'] = obj_feats
        feat_dict['union'] = union_feats

        return feat_dict
