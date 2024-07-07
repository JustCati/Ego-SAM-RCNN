import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def evaluate_one_epoch(model, loader, evaluator, tb_writer: SummaryWriter = None, epoch = 1):
    model.eval()
    device = model.device

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for _, target in enumerate(pbar, start=1):
            images, targets = target

            #* --------------- Forward Pass ----------------
            images = list([image.to(device) for image in images])
            targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]
            pred = model(images)

            #* --------------- Compute mAP ----------------
            img_size = (images[0].shape[-1], images[0].shape[-2])
            bbox_map, segm_map = evaluator.compute_map(pred, targets, img_size)

        #* --------------- Log mAP ----------------
        if tb_writer is not None:
            maps = {
                "bbox_map": bbox_map["map"],
                "segm_map": segm_map["map"]
            }
            tb_writer.add_scalars("val/map", maps, epoch)

        print("[Validation] Epoch: {:03d} Segmentation mAP: {:.2f}, Bounding Box mAP: {:.2f}".format(epoch, segm_map["map"], bbox_map["map"]))
        print()
    return bbox_map["map"], segm_map["map"]
