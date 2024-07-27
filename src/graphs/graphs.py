import json
import torch
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torchvision.transforms import transforms
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes



def plotSample(dataset):
    coco = dataset.coco
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
    labels = [coco.cats[label.item()]["name"] for label in labels]
    # labels = [metadata["categories"][label.item()]["name"] for label in labels]
    img = draw_bounding_boxes(img, boxes, labels=labels, colors="red", width=7, font_size=30)
    img = draw_segmentation_masks(img, targetMasks, alpha=0.3, colors="yellow")
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.title("Boxes & Masks")
    plt.tight_layout()
    plt.show()



def plotDemo(img, target, prediction, coco, save = False, path = None):
    plt.subplot(1, 2, 1)
    image = (img * 255).type(torch.uint8)
    targetMasks = target["masks"].type(torch.bool).reshape(-1, img.shape[-2], img.shape[-1])
    image = draw_segmentation_masks(image, targetMasks, alpha=0.5, colors="yellow")
    image = draw_bounding_boxes(image, target["boxes"], colors="green", width=3)
    plt.imshow(transforms.ToPILImage()(image), aspect='auto')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    img = (img * 255).type(torch.uint8)
    masks = prediction["masks"].type(torch.bool).reshape(-1, img.shape[-2], img.shape[-1])
    targetLabels = target["labels"]
    targetLabels = [coco.cats[label.item()]["name"] for label in targetLabels]
    predLabels = prediction["labels"]
    predLabels = [coco.cats[label.item()]["name"] for label in predLabels]
    img = draw_bounding_boxes(img, target["boxes"], labels=targetLabels, colors="green", width=3, font_size=30, font="Verdana.ttf")
    img = draw_bounding_boxes(img, prediction["boxes"], labels=predLabels, colors="red", width=3, font_size=30, font="Verdana.ttf")
    img = draw_segmentation_masks(img.type(torch.uint8), masks, alpha=0.5, colors="red")
    plt.imshow(transforms.ToPILImage()(img), aspect='auto')
    plt.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
