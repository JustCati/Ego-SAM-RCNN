import os
import json
import shutil
from pycocotools.coco import COCO



def unify_cocos(src_dir_path, dst_dir_path):
    dst_img_path = os.path.join(os.path.dirname(dst_dir_path), "images")
    if not os.path.exists(dst_img_path):
        os.makedirs(dst_img_path)

    newJson = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    id_counter, ann_counter, cat_counter = 1, 1, 1

    for folder in os.listdir(src_dir_path):
        print(f"Processing {folder}...")
        mapping = {
            "images": {},
            "annotations": {},
            "categories": {}
        }
        img_path = os.path.join(src_dir_path, folder, "val2017")
        ann_file = os.path.join(src_dir_path, folder, "annotations", "instances_val2017.json")

        coco = COCO(ann_file)
        for img in coco.dataset["images"]:
            new_id = id_counter
            mapping["images"][img["id"]] = new_id

            new_name = f"{id_counter:012}.jpg"
            new_path = os.path.join(dst_img_path, new_name)
            shutil.copyfile(os.path.join(img_path, img["file_name"]), new_path)

            new = img.copy()
            new["id"] = new_id
            new["file_name"] = new_name
            newJson["images"].append(new)
            id_counter += 1

        for cat in coco.dataset["categories"]:
            new_id = cat_counter
            mapping["categories"][cat["id"]] = new_id

            new = cat.copy()
            new["id"] = new_id
            newJson["categories"].append(new)
            cat_counter += 1

        for ann in coco.dataset["annotations"]:
            new_id = ann_counter
            mapping["annotations"][ann["id"]] = new_id

            new = ann.copy()
            new["id"] = new_id
            new["image_id"] = mapping["images"][ann["image_id"]]
            new["category_id"] = mapping["categories"][ann["category_id"]]
            newJson["annotations"].append(new)
            ann_counter += 1

    with open(dst_dir_path, "w") as f:
        json.dump(newJson, f)
