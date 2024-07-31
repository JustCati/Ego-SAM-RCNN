import os
import time
import argparse
import datetime
import os.path as osp

from src.model.model import getModel
from src.utils.checkpointer import Checkpointer

from src.engines.training import train

from src.dataset.create_masks import generate_masks
from src.dataset.coco import unify_cocos, split_coco
from src.utils.utils import get_device, fix_random_seed, worker_reset_seed

from src.utils.demo import demo
from src.dataset.dataloader import CocoDataset
from src.graphs.graphs import plotSample, plotDemo
from src.transform.transform import RandomGaussianBlur, GaussianNoise

import torch
from torch.utils import data
from torchvision.transforms import v2 as T
from torch.utils.tensorboard import SummaryWriter





def main(args):
    path = args.path
    if not osp.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")

    modelOutputPath = ""
    if args.train and not args.resume:
        modelOutputPath = osp.join(os.getcwd(), "ckpts", "sammask_rcnn_" + str(datetime.datetime.fromtimestamp(int(time.time()))))
        if not osp.exists(modelOutputPath):
            os.makedirs(modelOutputPath)
    elif args.resume:
        modelOutputPath = osp.dirname(args.resume)
    elif args.eval or args.demo:
        if args.eval:
            modelOutputPath = osp.dirname(args.eval)
        elif args.demo:
            modelOutputPath = osp.dirname(args.demo)

    if not osp.exists(modelOutputPath) and not args.sample:
        raise ValueError(f"Path {modelOutputPath} does not exist")


    #* Check if dataset annotations json file is already preprocessed
    cocoDirPath = osp.join(osp.dirname(path), "COCO")
    if not osp.exists(cocoDirPath):
        os.makedirs(cocoDirPath)
    annoPath = osp.join(cocoDirPath, "annotations")
    if not osp.exists(annoPath):
        os.makedirs(annoPath)

    if not osp.exists(osp.join(cocoDirPath, "ood_coco_unified.json")) and not osp.exists(osp.join(cocoDirPath, "images")):
        print("Unifying COCO datasets...")
        unify_cocos(path, osp.join(cocoDirPath, "annotations", "ood_coco_unified.json"))

    if not osp.exists(osp.join(annoPath, "ood_coco_unified_train.json")) and not osp.exists(osp.join(annoPath, "ood_coco_unified_eval.json")):
        print("Splitting COCO dataset...")
        split_coco(osp.join(annoPath, "ood_coco_unified.json"), annoPath)

    for split in ["eval", "train"]:
        #* Generate masks if necessary
        dst_coco = osp.join(annoPath, f"ood_coco_all_unified_{split}.json")
        if not osp.exists(dst_coco):
            device = get_device()
            sam_path = osp.join(os.getcwd(), "sam-checkpoints", "sam_vit_h.pth")
            print("Genereting masks for split: ", split)
            generate_masks(dst_coco.replace("_all", ""), osp.join(cocoDirPath, "images"), sam_path, device = device)


    img_path = osp.join(cocoDirPath, "images")
    valCocoPath = osp.join(annoPath, "ood_coco_all_unified_eval.json")
    trainCocoPath = osp.join(annoPath, "ood_coco_all_unified_train.json")

    #* --------------- Create Dataset -----------------

    if args.train or args.sample or args.demo or args.eval or args.resume != "":
        SEED = 1234567891
        rng_generator = fix_random_seed(SEED)

        #! Uncomment Gaussian Noise but performance will suffer a lot
        transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            RandomGaussianBlur(0.5, (5, 9), (0.1, 5)),
            # GaussianNoise(p = 0.5, noise_p = 0.07, mean = 0, sigma = 5),
        ])

        valSet = CocoDataset(img_path, valCocoPath)
        trainSet = CocoDataset(img_path, trainCocoPath, transforms = transform)

        trainDataloader = data.DataLoader(trainSet, 
                                        batch_size = args.batch_size, 
                                        num_workers = 8, 
                                        pin_memory = True, 
                                        shuffle = True,
                                        generator=rng_generator,
                                        worker_init_fn=worker_reset_seed,
                                        collate_fn = lambda x: tuple(zip(*x)))
        valDataloader = data.DataLoader(valSet, 
                                        batch_size = 1, 
                                        num_workers = 8, 
                                        pin_memory = True, 
                                        shuffle = False, 
                                        collate_fn = lambda x: tuple(zip(*x)))
        print("Dataset created")
        print("Number of classes: ", valSet.get_num_classes(), "+ 1 for background")
        print("Length of train dataloader: ", len(trainDataloader))
        print("Length of val dataloader: ", len(valDataloader))

    if args.sample:
        plotSample(valSet)
        exit()

    #* ----------------------------------------------------


    #* --------------- Train the model -----------------

    curr_epoch = 0
    num_classes = valSet.get_num_classes()
    EPOCHS = args.epochs if args.epochs > 0 else 10
    tb_writer = SummaryWriter(osp.join(modelOutputPath, "logs"))

    device = get_device()
    model = getModel(num_classes)
    model.to(device)

    if args.train or args.resume != "":
        print("\nTraining model")

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1)
        checkpointer = Checkpointer(args.resume if args.resume != "" else modelOutputPath, phase = 'train')

        if args.resume != "" and osp.exists(args.resume) and osp.isfile(args.resume):
            print("Most recent trained model found, continuing training...")
            model, optimizer, lr_scheduler, curr_epoch = checkpointer.load(model, optimizer, lr_scheduler)
            model.to(device)

        cfg = {
            "model" : model,
            "optimizer" : optimizer,
            "lr_scheduler" : lr_scheduler,
            "curr_epoch" : curr_epoch,
            "epoch" : curr_epoch + (EPOCHS - curr_epoch),
            "trainDataloader" : trainDataloader,
            "valDataloader" : valDataloader,
            "tb_writer" : tb_writer,
            "checkpointer" : checkpointer,
            "device" : device
        }
        train(cfg)
    elif args.eval != "":
        if osp.exists(osp.join(modelOutputPath, args.eval)):
            model, *_ = Checkpointer(modelOutputPath, phase = 'eval').load(model, None, None)
            model.to(device)
    elif args.demo != "":
        if osp.exists(args.demo):
            print("Model found, loading...")
            model, *_ = Checkpointer(args.demo, phase = 'eval').load(model, None, None)
            model.to(device)
    else:
        raise ValueError("No model checkpoint found")

    #* --------------- Plot inferenced example -----------------
    if args.demo:
        MASK_THRESHOLD = 0.5
        if not args.save:
            for _ in range(3):
                (img, target) = valSet[torch.randint(0, len(valSet), (1,))]

                cfg = {
                    "model" : model,
                    "img" : img,
                    "target" : target,
                    "MASK_THRESHOLD" : MASK_THRESHOLD,
                    "device" : device
                }
                pred = demo(**cfg)
                if pred["prediction"]["boxes"].shape[0] == 0:
                    print(f"No predictions on img {valSet.coco.imgs[target['image_id'].item()]['file_name']}")
                    continue
                plotDemo(**pred, coco = valSet.coco)
                del pred
                del cfg
        else:
            demoPath = osp.join(modelOutputPath, "demo")
            if not osp.exists(demoPath):
                os.makedirs(demoPath)
            for idx, (img, target) in enumerate(valSet):
                cfg = {
                    "model" : model,
                    "img" : img,
                    "target" : target,
                    "MASK_THRESHOLD" : MASK_THRESHOLD,
                    "device" : device
                }
                pred = demo(**cfg)
                outPath = osp.join(demoPath, f"demo_{target["image_id"].item()}.png")
                plotDemo(**pred, coco = valSet.coco, save=True, path=outPath)
                del pred
                del cfg
    #* ----------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coco-O(ut of distribution) Istance Segmentation with SAMASK-RCNN")
    parser.add_argument("--path", type=str, default=osp.join(os.getcwd(), "data", "OOD-COCO", "OOD-COCO"), help="Path to the data directory")
    parser.add_argument("--sample", action="store_true", default=False, help="Plot a sample image from the dataset with ground truth masks")
    parser.add_argument("--train", action="store_true", default=False, help="Force Training of the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--demo", type=str, default="", help="Run a demo of inference on 3 random image from the validation set with the model checkpoint at the specified path")
    parser.add_argument("--eval", type=str, default="", help="Evaluate the model checkpoint at the specified path")
    parser.add_argument("--save", action="store_true", default=False, help="Save the demo images to the model directory")
    parser.add_argument("--resume", type=str, default="", help="Resume training from the specified model checkpoint path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    args = parser.parse_args()

    main(args)
