import torch.nn as nn
from torchvision.ops import nms

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def getModel(num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights = "DEFAULT", backbone_weights = "DEFAULT", box_score_thresh = 0.5)

    #* Change the number of output classes
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels = in_features_box, 
        num_classes = num_classes + 1
    )
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels = in_features_mask, 
        dim_reduced = dim_reduced, 
        num_classes = num_classes + 1
    )
    return model
