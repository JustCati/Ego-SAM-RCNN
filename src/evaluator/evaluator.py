from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



class Evaluator():
    def __init__(self, 
                 gt: COCO,
                 pred_box_json: str,
                 pred_mask_json: str):
        self.cocoGT = gt
        self.pred_box_json = pred_box_json
        self.pred_mask_json = pred_mask_json

        try: 
            self.coco_box = self.cocoGT.loadRes(self.pred_box_json)
        except:
            self.coco_box = None
        try:
            self.coco_mask = self.cocoGT.loadRes(self.pred_mask_json)
        except:
            self.coco_mask = None


    #* Single wrapper function for memory reason
    #* (Python deletes the objects only if they are in differents functions)
    def single_compute(self, type: str = "bbox"):
        if type == "bbox":
            cocoeval = COCOeval(self.cocoGT, self.coco_box, "bbox")
        elif type == "segm":
            cocoeval = COCOeval(self.cocoGT, self.coco_mask, "segm")

        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()
        return cocoeval.stats


    def compute_map(self):
        box_map = self.single_compute("bbox") if self.coco_box is not None else [0.0]
        mask_map = self.single_compute("segm") if self.coco_mask is not None else [0.0]
        return box_map, mask_map
