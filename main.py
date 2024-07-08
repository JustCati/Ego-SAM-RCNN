import os
import time
import argparse
import datetime

from src.model.model import MaskRCNN
from src.evaluator.evaluator import Evaluator
from src.utils.checkpointer import Checkpointer

from src.engines.training import train

from src.dataset.coco import convert_to_coco
from src.dataset.create_masks import generate_masks
from src.utils.utils import get_device, fix_random_seed, worker_reset_seed

from src.graphs.graphs import plotSample
from src.dataset.dataloader import CocoDataset
from src.transform.transform import RandomGaussianBlur, GaussianNoise

import torch
from torch.utils import data
from torchvision.transforms import v2 as T
from torch.utils.tensorboard import SummaryWriter





def main(args):
    path = args.path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")

    img_path = os.path.join(path, "images")
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    modelOutputPath = ""
    if args.train:
        modelOutputPath = os.path.join(os.getcwd(), "ckpts", "sammask_rcnn_" + str(datetime.datetime.fromtimestamp(int(time.time()))))
        if not os.path.exists(modelOutputPath):
            os.makedirs(modelOutputPath)
    elif args.resume:
        modelOutputPath = os.path.dirname(args.resume)
    elif args.perf or args.eval or args.demo:
        if args.perf:
            modelOutputPath = os.path.dirname(args.perf)
        elif args.eval:
            modelOutputPath = os.path.dirname(args.eval)
        elif args.demo:
            modelOutputPath = os.path.dirname(args.demo)

    if not os.path.exists(modelOutputPath) and not args.sample:
        raise ValueError(f"Path {modelOutputPath} does not exist")


    #* Check if dataset annotations json file is already preprocessed
    device = get_device()
    annoPath = os.path.join(path, "annotations")
    cocoPath = os.path.join(os.path.dirname(path), "COCO")
    metadataPath = os.path.join(annoPath, "ego_objects_metadata.json")

    for split in ["eval", "train"]:
        #* Convert to COCO if necessary
        src_json = os.path.join(annoPath, f"ego_objects_{split}.json")
        dst_coco = os.path.join(cocoPath, f"ego_objects_coco_{split}.json")

        if not os.path.exists(cocoPath):
            os.makedirs(cocoPath)
        if not os.path.exists(dst_coco):
            print("Converting to COCO format for split: ", split)
            convert_to_coco(src_json, metadataPath, dst_coco)

        #* Generate masks if necessary
        if not os.path.exists(dst_coco.replace("coco", "coco_all")):
            sam_path = os.path.join(os.getcwd(), "sam-checkpoints", "sam_vit_h.pth")
            print("Genereting masks for split: ", split)
            generate_masks(dst_coco, img_path, sam_path, device = device)

    valCocoPath = os.path.join(cocoPath, "ego_objects_coco_all_eval.json")
    trainCocoPath = os.path.join(cocoPath, "ego_objects_coco_all_train.json")

    #* --------------- Create Dataset -----------------

    if args.train or args.sample or args.demo or args.eval or args.resume != "":
        SEED = 1234567891
        rng_generator = fix_random_seed(SEED)

        #! Uncomment Gaussian Noise but performance will suffer a lot
        transform = T.Compose([
            T.Resize(640),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            RandomGaussianBlur(0.5, (5, 9), (0.1, 5)),
            # GaussianNoise(p = 0.5, noise_p = 0.07, mean = 0, sigma = 5),
        ])

        valSet = CocoDataset(img_path, valCocoPath, transform=T.Compose([T.Resize(640)]))
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
        print("Length of train dataloader: ", len(trainDataloader))
        print("Length of val dataloader: ", len(valDataloader))

    if args.sample:
        plotSample(valSet, metadataPath)

    #* ----------------------------------------------------


    #* --------------- Train the model -----------------

    if args.train:
        curr_epoch = 1
        EPOCHS = args.epochs + 1 if args.epochs > 0 else 10
        tb_writer = SummaryWriter(os.path.join(modelOutputPath, "logs"))

        num_classes = valSet.get_num_classes()
        thresholds = torch.arange(0.5, 0.95, 0.05).tolist()
        evaluator = Evaluator(bbox_metric = "map", segm_metric = "map", thresholds=thresholds)

        device = get_device()
        model = MaskRCNN(num_classes, pretrained = True, weights = "DEFAULT", backbone_weights = "DEFAULT")
        model.to(device)

        if args.train or args.resume != "":
            print("\nTraining model")

            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.001)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
            checkpointer = Checkpointer(args.resume if args.resume != "" else modelOutputPath, phase = 'train')

            if args.resume != "" and os.path.exists(args.resume) and os.path.isfile(args.resume):
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
                "evaluator" : evaluator
            }
            train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EgoObject Istance Segmentation with SAMASK-RCNN")
    parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "data", "EgoObjects"), help="Path to the data directory")
    parser.add_argument("--sample", action="store_true", default=False, help="Plot a sample image from the dataset with ground truth masks")
    parser.add_argument("--train", action="store_true", default=False, help="Force Training of the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--demo", type=str, default="", help="Run a demo of inference on 3 random image from the validation set with the model checkpoint at the specified path")
    parser.add_argument("--perf", type=str, default="", help="Plot the performance of the model checkpoint at the specified path")
    parser.add_argument("--eval", type=str, default="", help="Evaluate the model checkpoint at the specified path")
    parser.add_argument("--save", action="store_true", default=False, help="Save the demo images to the model directory")
    parser.add_argument("--resume", type=str, default="", help="Resume training from the specified model checkpoint path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    args = parser.parse_args()

    main(args)
