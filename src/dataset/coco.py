import json
from lvis import LVIS



# Convert to COCO format
def convert_to_coco(src_json_path, dst_path):
    lvis = LVIS(src_json_path)    
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

    with open(dst_path, "w") as f:
        json.dump(coco, f)
