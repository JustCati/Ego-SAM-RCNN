import os
import sys
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.mask import encode
from torchvision.ops.boxes import box_convert
sys.path.append(os.path.join(os.getcwd(), "sam", "segment_anything"))

import torch
from sam.segment_anything import (
    sam_model_registry, 
    SamPredictor
)
import warnings
warnings.filterwarnings("ignore")



def generate_masks(cocoPath, imgPath, sam_path = None, device = "cuda"):
    if not os.path.exists(cocoPath):
        raise ValueError(f"Path {cocoPath} does not exist")

    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    coco = COCO(cocoPath)
    with open(cocoPath, "r") as f:
        cocoJSON = json.load(f)
    annMap = {ann["id"]: i for i, ann in enumerate(cocoJSON["annotations"])}
    
    img_ids = coco.getImgIds()
    for img_id in tqdm(img_ids):
        img = coco.loadImgs(img_id)[0]
        img_path = os.path.join(imgPath, img["file_name"])
        img_shape = (img["height"], img["width"])

        # Load image
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Load bboxes and convert them from xywh to xyxy
        boxes = []
        anns_ids = coco.getAnnIds(imgIds=img_id)
        for ann in coco.loadAnns(anns_ids):
            boxes.append(ann["bbox"])
        boxes = box_convert(torch.tensor(boxes), in_fmt="xywh", out_fmt="xyxy")
        boxes = torch.tensor(boxes).type(torch.int64).to(predictor.device)

        # Load and convert data for SAM
        sam_boxes = predictor.transform.apply_boxes_torch(boxes, img_shape)
        predictor.set_image(img)
        masks, *_ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=sam_boxes,
            multimask_output=False
            )
        masks = masks.reshape(-1, *img_shape)

        # Convert binary mask to RLE and update the COCO JSON
        for i, mask in enumerate(masks):
            mask = mask.cpu().numpy()
            mask = np.asfortranarray(mask)
            ann_idx = annMap[anns_ids[i]]
            rle = encode(mask)
            rle["counts"] = rle["counts"].decode("utf-8")
            cocoJSON["annotations"][ann_idx]["segmentation"] = rle

    with open(cocoPath.replace("coco", "coco_all"), "w") as f:
        json.dump(cocoJSON, f)
