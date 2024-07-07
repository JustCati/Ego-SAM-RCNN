import torch
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision as map



class Evaluator():
    def __init__(self, 
                 bbox_metric: str = "map", 
                 segm_metric: str = "map", 
                 thresholds: list = torch.arange(0.5, 0.95, 0.05).tolist()):
        self.bbox_metric = bbox_metric
        self.segm_metric = segm_metric
        self.thresholds = thresholds

        self.map_bbox = map(iou_thresholds = self.thresholds,
                            box_format="xyxy",
                            iou_type="bbox",
                            backend="faster_coco_eval")
        self.segm_mask = map(iou_thresholds = self.thresholds,
                            iou_type="segm",
                            backend="faster_coco_eval")


    def compute_map(self, preds, targets):
        self.map_bbox.update(preds, targets)
        bbox_map = self.map_bbox.compute()

        segm_maps = []
        targets = [{k: v.reshape(-1, v.shape[-2], v.shape[-1])
                        if k == "masks" else v for k, v in elem.items()} for elem in targets]

        for th in self.thresholds:
            act_preds = [{k: (v > th).reshape(-1, v.shape[-2], v.shape[-1]) 
                            if k == "masks" else v for k, v in elem.items()} for elem in preds]
            self.segm_mask.update(act_preds, targets)
            segm_maps.append(self.segm_mask.compute())
        segm_map = {k: np.mean([elem[k] for elem in segm_maps]) for k in segm_maps[0].keys()}

        return bbox_map, segm_map
