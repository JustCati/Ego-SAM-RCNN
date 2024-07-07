from torchvision.ops import nms

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




class MaskRCNN():
    def __init__(self, num_classes, pretrained = True, weights = "DEFAULT", backbone_weights = "DEFAULT"):
        self.device = None
        if pretrained:
            self.model = maskrcnn_resnet50_fpn_v2(weights = weights, backbone_weights = backbone_weights)
        else:
            self.model = maskrcnn_resnet50_fpn_v2()

        #* Change the number of output classes
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = self.model.roi_heads.mask_predictor.conv5_mask.out_channels

        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels = in_features_box, 
            num_classes = num_classes
        )
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels = in_features_mask, 
            dim_reduced = dim_reduced, 
            num_classes = num_classes
        )

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device)

    def parameters(self):
        return self.model.parameters()

    #TODO: Change the forward method to implement nms
    def forward(self, x, y = None):
        return self.model(x, y)
