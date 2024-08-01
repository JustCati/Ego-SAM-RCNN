import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    img = draw_bounding_boxes(img, boxes, labels=labels, colors="red", width=7, font_size=15, font="Verdana.ttf")
    img = draw_segmentation_masks(img, targetMasks, alpha=0.3, colors="yellow")
    img = img.permute(1, 2, 0)
    plt.imshow(img)
    plt.title("Boxes & Masks")
    plt.tight_layout()
    plt.show()



def plotDemo(img, target, prediction, coco, save=False, path=None):
    def draw_boxes(image, boxes, labels=None, scores=None, coco=None):
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if labels is not None:
                class_id = labels[i].item()
                label = coco.cats[class_id]['name']
                if scores is not None:
                    score = scores[i].item()
                    text = f"{label}: {score:.2f}"
                else:
                    text = label
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.rectangle(image, (x1, y1 - text_height - 2), (x1 + text_width, y1), (0, 0, 0), -1)
                cv2.putText(image, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image

    def draw_masks(image, masks):
        image = torch.tensor(image).permute(2, 0, 1)
        masks = masks.type(torch.bool).reshape(-1, image.shape[-2], image.shape[-1])
        image = draw_segmentation_masks(image, masks, alpha=0.5, colors="green")
        return image.permute(1, 2, 0).cpu().numpy()

    image_np = img.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    gt_image = image_np.copy()
    pred_image = image_np.copy()

    gt_boxes = target['boxes']
    gt_labels = target['labels']
    gt_image = draw_boxes(gt_image, gt_boxes, gt_labels, coco=coco)
    gt_image = draw_masks(gt_image, target['masks'])

    pred_boxes = prediction['boxes']
    pred_labels = prediction['labels']
    pred_scores = prediction['scores']
    pred_image = draw_boxes(pred_image, pred_boxes, pred_labels, pred_scores, coco=coco)
    pred_image = draw_masks(pred_image, prediction['masks'])

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(gt_image)
    axs[0].set_title('Ground Truth')
    axs[0].axis('off')

    axs[1].imshow(pred_image)
    axs[1].set_title('Prediction')
    axs[1].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
