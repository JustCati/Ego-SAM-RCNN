import os
import torch
from tqdm import tqdm
from .evaluating import evaluate_one_epoch

from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, loader, optimizer, lr_scheduler, tb_writer: SummaryWriter, epoch):
    model.train()
    device = model.device
    num_iters = len(loader)

    pbar = tqdm(loader, desc=f"Training epoch {epoch + 1}")
    for iter, target in enumerate(pbar):
        #* --------------- Forward Pass ----------------
        images, targets = target
        images = list([image.to(device) for image in images])
        targets = [{k: v.to(device) for k, v in elem.items()} for elem in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        #* --------------- Log Losses to Tensorboard ----------------
        global_step = epoch * num_iters + iter

        losses = {
            "loss_box_reg": loss_dict["loss_box_reg"].item(), 
            "loss_mask": loss_dict["loss_mask"].item(),
            "loss_classifier": loss_dict["loss_classifier"].item(),
        }
        tb_writer.add_scalars("train/all_losses", losses, global_step)
        tb_writer.add_scalar("train/final_loss", loss.item() / len(images), global_step)
        
        #* --------------- Log Learning Rate to Tensorboard ----------------
        tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        #* --------------- Log Progress to pbar ----------------
        pbar.set_postfix({
            'lr': '{:.7f}'.format(lr_scheduler.get_last_lr()[0]),
            "cls_loss": '{:.5f}'.format(loss_dict["loss_classifier"].item()),
            'box_loss': '{:.5f}'.format(loss_dict["loss_box_reg"].item()),
            'mask_loss': '{:.5f}'.format(loss_dict["loss_mask"].item())
        })

        #* --------------- Backward and Optimize ----------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        torch.cuda.empty_cache()
    return



def train(cfg):

    #* --------------- Load Config ----------------
    model = cfg["model"]
    optimizer = cfg["optimizer"]
    lr_scheduler = cfg["lr_scheduler"]
    curr_epoch = cfg["curr_epoch"]
    n_epoch = cfg["epoch"]
    trainLoader = cfg["trainDataloader"]
    valLoader = cfg["valDataloader"]
    tb_writer = cfg["tb_writer"]
    checkpointer = cfg["checkpointer"]
    #* --------------------------------------------

    cocoGT = valLoader.dataset.coco
    predPath = os.path.join(os.path.dirname(valLoader.dataset.annfile), "results.json")

    #* --------------- Train and Evaluate ----------------
    print("\nStart training model...")

    for epoch in range(curr_epoch, n_epoch):
        train_one_epoch(model, 
                        trainLoader,
                        optimizer,
                        lr_scheduler,
                        tb_writer,
                        epoch)
        torch.cuda.empty_cache()
        bbox_map, segm_map = evaluate_one_epoch(model,
                            valLoader,
                            cocoGT,
                            predPath,
                            tb_writer,
                            epoch)

        checkpointer.save(epoch + 1, model, optimizer, lr_scheduler, bbox_map, segm_map)

    #* ----------------------------------------------

    print("Training completed")
    return
