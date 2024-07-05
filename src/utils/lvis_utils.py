import json


def swap_categories_ids(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    for i in range(len(data["annotations"])):
        data["annotations"][i]["category_id"] = data["annotations"][i]["_category_id"]
    with open(json_path, "w") as f:
        json.dump(data, f)


def fix_annotations(split_path, metadata_path, split="train"):
    outputPath = split_path.replace(".json", "_fixed.json")
    swap_categories_ids(outputPath)
