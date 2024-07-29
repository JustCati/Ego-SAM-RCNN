import os
import torch
import json
import numpy as np
from tqdm import tqdm
from pycocotools.mask import encode
from torchvision.ops import box_convert
from torch.utils.tensorboard import SummaryWriter

from src.evaluator.evaluator import Evaluator



def evaluate_one_epoch(model, loader, cocoGT, predPath, tb_writer: SummaryWriter = None, epoch = -1, device = "cpu"):
    model.eval()
    box_results, mask_results = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validating epoch {epoch + 1}")
        for _, target in enumerate(pbar, start=1):
            images, targets = target
            img_ids = [elem["image_id"] for elem in targets]
            del targets

            #* --------------- Forward Pass ----------------
            try:
                images = list([image.to(device) for image in images])
                pred = model(images)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory error detected. Retrying...")
                    print(f"Skipping image ids: {img_ids}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            #* --------------- Create Prediction File ----------------
            for i, elem in enumerate(pred):
                for idx, bbox in enumerate(elem["boxes"]):
                   box_results.append({
                          "image_id": img_ids[i].item(),
                          "category_id": elem["labels"][idx].item(),
                          "bbox": [round(elem, 2) for elem in box_convert(bbox, "xyxy", "xywh").tolist()],
                          "score": round(elem["scores"][idx].item(), 2)
                     })

                masks = elem["masks"].squeeze(1)
                for idx, mask in enumerate(masks):
                    toMean = mask[torch.where(mask > 0.0)]
                    score = torch.mean(toMean).item()
                    mask = encode(np.asfortranarray((mask > 0.5).cpu().numpy()))
                    mask["counts"] = mask["counts"].decode("utf-8")

                    mask_results.append({
                        "image_id": img_ids[i].item(),
                        "category_id": elem["labels"][idx].item(),
                        "segmentation": mask,
                        "score": round(score, 2)
                    })
            torch.cuda.empty_cache()

        #* --------------- Save Prediction File ----------------

        output_box_path = predPath.replace("results", "box_results")
        output_mask_path = predPath.replace("results", "mask_results")

        #* Prefer to save the results in a file instead of keeping them in memory
        #* to avoid memory issues with coco evaluator
        with open(output_box_path, "w") as f:
            json.dump(box_results, f)
        with open(output_mask_path, "w") as f:
            json.dump(mask_results, f)
        del box_results, mask_results

        #* --------------- Evaluate ----------------
        evaluator = Evaluator(cocoGT, output_box_path, output_mask_path)
        bbox_map, segm_map = evaluator.compute_map()
        os.remove(output_box_path)
        os.remove(output_mask_path)

        #* --------------- Log mAP ----------------

        if tb_writer is not None:
            maps = {
                "bbox_map": bbox_map[0],
                "segm_map": segm_map[0]
            }
            tb_writer.add_scalars("val/map", maps, epoch + 1)

        print("[Validation] Epoch: {:03d} Segmentation mAP: {:.2f}, Bounding Box mAP: {:.2f}".format(epoch + 1, segm_map[0], bbox_map[0]))
        print()
    return bbox_map[0], segm_map[0]
