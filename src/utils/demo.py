import os
import json
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.mask import encode
from torchvision.ops import box_convert
from src.evaluator.evaluator import Evaluator


def demo(model, img, target, MASK_THRESHOLD, device):
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        prediction = {k: v.to("cpu") for k, v in prediction[0].items()}

    for i in range(len(prediction["masks"])):
        prediction["masks"][i] = prediction["masks"][i] > MASK_THRESHOLD

    return {
        "img": img,
        "target": target,
        "prediction": prediction
    }



def evaluate(model, dataloader, device):
    model.eval()
    box_results, mask_results = [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validating epoch -1")
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

        #* --------------- Evaluate ----------------
        evaluator = Evaluator(dataloader.dataset.coco, box_results, mask_results)
        evaluator.compute_map()
    return
