import os
import json
import shutil
import random
import os.path as osp
from pycocotools.coco import COCO



def unify_cocos(src_dir_path, dst_dir_path):
    dst_img_path = osp.join(osp.dirname(osp.dirname(dst_dir_path)), "images")
    if not osp.exists(dst_img_path):
        os.makedirs(dst_img_path)

    newJson = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    cat = False
    id_counter, ann_counter = 1, 1

    for folder in os.listdir(src_dir_path):
        # Check if folder is named "images"
        if folder == "images":
            continue
        print(f"Processing {folder}...")
        mapping = {
            "images": {},
            "annotations": {}
        }
        img_path = osp.join(src_dir_path, folder, "val2017")
        ann_file = osp.join(src_dir_path, folder, "annotations", "instances_val2017.json")

        if not osp.exists(ann_file):
            print(f"Annotation file not found: {ann_file}")
            continue

        coco = COCO(ann_file)
        if not cat:
            newJson["categories"] = coco.dataset["categories"]
            cat = True

        # Process images
        for img in coco.dataset["images"]:
            new_id = id_counter
            mapping["images"][img["id"]] = new_id

            new_name = f"{id_counter:012}.jpg"
            new_path = osp.join(dst_img_path, new_name)

            src_img_path = osp.join(img_path, img["file_name"])
            if not osp.exists(src_img_path):
                print(f"Image file not found: {src_img_path}")
                continue

            shutil.copyfile(src_img_path, new_path)

            new = img.copy()
            new["id"] = new_id
            new["file_name"] = new_name
            newJson["images"].append(new)
            id_counter += 1

        # Process annotations
        for ann in coco.dataset["annotations"]:
            if ann["image_id"] not in mapping["images"]:
                print(f"Annotation with image_id {ann['image_id']} is missing in the images list.")
                continue
            # if ann["category_id"] not in mapping["categories"]:
            #     print(f"Annotation with category_id {ann['category_id']} is missing in the categories list.")
            #     continue

            new_id = ann_counter
            mapping["annotations"][ann["id"]] = new_id

            new = ann.copy()
            new["id"] = new_id
            new["image_id"] = mapping["images"][ann["image_id"]]
            new["category_id"] = ann["category_id"]
            newJson["annotations"].append(new)
            ann_counter += 1

    
    with open(dst_dir_path, "w") as f:
        json.dump(newJson, f, indent=4)


def split_coco(src_json, dst_dir):
    coco = COCO(src_json)
    length = len(coco.dataset["images"])
    perc_75 = int(length * 0.75)
    indices = list(range(length))

    random.Random(4).shuffle(indices)
    train_indices = indices[:perc_75]
    val_indices = indices[perc_75:]

    train_json = {
        "images": [],
        "annotations": [],
        "categories": coco.dataset["categories"]
    }
    val_json = {
        "images": [],
        "annotations": [],
        "categories": coco.dataset["categories"]
    }

    for idx in train_indices:
        img = coco.dataset["images"][idx]
        train_json["images"].append(img)
        if img["id"] in coco.imgToAnns:
            for ann in coco.imgToAnns[img["id"]]:
                train_json["annotations"].append(ann)

    for idx in val_indices:
        img = coco.dataset["images"][idx]
        val_json["images"].append(img)
        if img["id"] in coco.imgToAnns:
            for ann in coco.imgToAnns[img["id"]]:
                val_json["annotations"].append(ann)

    name = osp.basename(src_json).split(".")[0].replace("_all", "")
    with open(osp.join(dst_dir, f"{name}_train.json"), "w") as f:
        json.dump(train_json, f, indent=4)
    with open(osp.join(dst_dir, f"{name}_eval.json"), "w") as f:
        json.dump(val_json, f, indent=4)
