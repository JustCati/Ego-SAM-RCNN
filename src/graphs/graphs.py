import torch
import json
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

import numpy as np
import matplotlib.pyplot as plt



def plotSample(dataset, metadata):
    with open(metadata, "r") as f:
        metadata = json.load(f)
    (img, target) = dataset[torch.randint(0, len(dataset), (1,))]

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    img_mask = torch.zeros((3, img.shape[-2], img.shape[-1]), dtype=torch.uint8)
    targetMasks = target["masks"].type(torch.bool).reshape(-1, img.shape[-2], img.shape[-1])
    img_mask = draw_segmentation_masks(img_mask, targetMasks, alpha=0.5, colors="yellow")
    img_mask = img_mask.permute(1, 2, 0)
    plt.imshow(img_mask)
    plt.title("Masks")

    plt.subplot(1, 2, 2)
    plt.axis('off')
    img = (img * 255).type(torch.uint8)
    boxes = target["boxes"].reshape(-1, 4)
    labels = target["labels"]
    labels = [metadata["categories"][label.item()]["name"] for label in labels]
    img = draw_bounding_boxes(img, boxes, labels=labels, colors="red", width=7)
    img = draw_segmentation_masks(img, targetMasks, alpha=0.3, colors="yellow")
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.title("Boxes & Masks")
    plt.tight_layout()
    plt.show()
