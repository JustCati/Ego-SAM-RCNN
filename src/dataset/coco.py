import os
import json
from lvis import LVIS



def swap_categories_ids(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    for i in range(len(data["annotations"])):
        data["annotations"][i]["category_id"] = data["annotations"][i]["_category_id"]
    with open(json_path.replace(".json", "_fixed.json"), "w") as f:
        json.dump(data, f)
    return json_path.replace(".json", "_fixed.json")


def append_metadata(json_path, metadata_json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    with open(metadata_json_path, "r") as f:
        metadata = json.load(f)
    data["categories"] = metadata["categories"]
    with open(json_path, "w") as f:
        json.dump(data, f)



# Convert to COCO format
def convert_to_coco(src_json_path, metadata_json_path, dst_path):

    # Copy _category_id in category_id for each annotation
    print("Swapping categories ids...")
    new_path = swap_categories_ids(src_json_path)

    # Append categories ids to metadata
    print("Appending categories ids to metadata...")
    append_metadata(new_path, metadata_json_path)


    lvis = LVIS(new_path)
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Copy categories
    for cat in lvis.load_cats(None):
        coco["categories"].append({
            "id": cat["id"],
            "name": cat["name"]
        })
    print("Categories: ", len(coco["categories"]))
    
    # Copy annotations
    for ann in lvis.load_anns(None):
        coco["annotations"].append({
            "id": ann["id"],
            "image_id": ann["image_id"],
            "bbox": ann["bbox"],
            "bbox_mode": "XYWH",
            "iscrowd": 0,
            "area": ann["area"],
            "category_id": ann["category_id"],
        })
    print("Annotations: ", len(coco["annotations"]))

    # Copy images metadata
    for img in lvis.load_imgs(None):
        coco["images"].append({
            "id": img["id"],
            "file_name": img["url"],
            "height": img["height"],
            "width": img["width"],
        })
    print("Images: ", len(coco["images"]))
    print()

    os.remove(new_path)
    with open(dst_path, "w") as f:
        json.dump(coco, f)
