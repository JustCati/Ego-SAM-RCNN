import os
import sys
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
sys.path.append(os.path.join(os.getcwd(), "sam", "segment_anything"))

import torch
from sam.segment_anything import (
    sam_model_registry, 
    SamPredictor
)
import warnings
warnings.filterwarnings("ignore")



def binary_mask_to_rle_np(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()
    return rle



def generate_masks(cocoPath, imgPath, sam_path = None):
    if not os.path.exists(cocoPath):
        raise ValueError(f"Path {cocoPath} does not exist")

    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to(torch.device("cuda"))
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

        boxes = []
        anns_ids = coco.getAnnIds(imgIds=img_id)
        for ann in coco.loadAnns(anns_ids):
            boxes.append(ann["bbox"])
        boxes = torch.tensor(boxes).type(torch.int64).to(predictor.device)
        predictor.transform.apply_boxes_torch(boxes, img_shape)

        predictor.set_image(img)
        masks, *_ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=False
            )

        for i, mask in enumerate(masks):
            mask = mask.cpu().numpy().astype(np.uint8)
            rle = binary_mask_to_rle_np(mask)

            ann_idx = annMap[anns_ids[i]]
            cocoJSON["annotations"][ann_idx]["segmentation"] = {
                "counts": rle["counts"], 
                "size": rle["size"]
            }

    with open(cocoPath, "w") as f:
        json.dump(cocoJSON, f)
