import os
import time
import json
import argparse
import datetime

from src.dataset.coco import convert_to_coco
from src.dataset.create_masks import generate_masks
from src.utils.utils import get_device, fix_random_seed, worker_reset_seed

from src.graphs.graphs import plotSample
from src.dataset.dataloader import CocoDataset

import torch
from torch.utils import data
from torchvision.transforms import v2 as T
from src.transform.transform import RandomGaussianBlur, GaussianNoise





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
    metadataPath = os.path.join(annoPath, "ego_objects_metadata.json")

    num_classes = 0
    for split in ["eval", "train"]:
        #* Convert to COCO if necessary
        src_json = os.path.join(annoPath, f"ego_objects_{split}.json")
        cocoPath = os.path.join(path, "COCO", f"ego_objects_coco_{split}.json")

        if not os.path.exists(os.path.join(path, "COCO")):
            os.makedirs(os.path.join(path, "COCO"))
        if not os.path.exists(cocoPath):
            print("Converting to COCO format for split: ", split)
            num_classes = convert_to_coco(src_json, metadataPath, cocoPath)

        #* Generate masks if necessary
        if not os.path.exists(cocoPath.replace("coco", "coco_all")):
            sam_path = os.path.join(os.getcwd(), "sam-checkpoints", "sam_vit_h.pth")
            print("Genereting masks for split: ", split)
            generate_masks(cocoPath, img_path, sam_path, device = device)

    valCocoPath = os.path.join(path, "COCO", "ego_objects_coco_all_eval.json")
    trainCocoPath = os.path.join(path, "COCO", "ego_objects_coco_all_train.json")

    #* --------------- Create Dataset -----------------

    if args.train or args.sample or args.demo or args.eval or args.resume != "":
        SEED = 1234567891
        rng_generator = fix_random_seed(SEED)

        #! Uncomment Gaussian Noise but performance will suffer a lot
        transform = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(degrees = (0, 180)),
            T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            RandomGaussianBlur(0.5, (5, 9), (0.1, 5)),
            # GaussianNoise(p = 0.5, noise_p = 0.07, mean = 0, sigma = 5),
        ])

        val = CocoDataset(img_path, valCocoPath)
        train = CocoDataset(img_path, trainCocoPath, transforms = transform)

        BATCH_SIZE = 3
        trainDataloader = data.DataLoader(train, 
                                        batch_size = BATCH_SIZE, 
                                        num_workers = 8, 
                                        pin_memory = True, 
                                        shuffle = True,
                                        generator=rng_generator,
                                        worker_init_fn=worker_reset_seed,
                                        collate_fn = lambda x: tuple(zip(*x)))
        valDataloader = data.DataLoader(val, 
                                        batch_size = 1, 
                                        num_workers = 8, 
                                        pin_memory = True, 
                                        shuffle = False, 
                                        collate_fn = lambda x: tuple(zip(*x)))

    if args.sample:
        plotSample(val, metadataPath)

    #* ----------------------------------------------------



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EgoObject Istance Segmentation with SAMASK-RCNN")
    parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "data", "EgoObjects"), help="Path to the data directory")
    parser.add_argument("--sample", action="store_true", default=False, help="Plot a sample image from the dataset with ground truth masks")
    parser.add_argument("--train", action="store_true", default=False, help="Force Training of the model")
    parser.add_argument("--demo", type=str, default="", help="Run a demo of inference on 3 random image from the validation set with the model checkpoint at the specified path")
    parser.add_argument("--perf", type=str, default="", help="Plot the performance of the model checkpoint at the specified path")
    parser.add_argument("--eval", type=str, default="", help="Evaluate the model checkpoint at the specified path")
    parser.add_argument("--save", action="store_true", default=False, help="Save the demo images to the model directory")
    parser.add_argument("--resume", type=str, default="", help="Resume training from the specified model checkpoint path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    args = parser.parse_args()

    main(args)
