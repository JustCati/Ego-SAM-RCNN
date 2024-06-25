import json


def append_categories(split_path, metadata_path, outPath):
    with open(split_path, "r") as f:
        eval = json.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    eval["categories"] = metadata["categories"]
    with open(outPath, "w") as f:
        json.dump(eval, f)


def swap_categories_ids(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    for i in range(len(data["annotations"])):
        data["annotations"][i]["category_id"] = data["annotations"][i]["_category_id"]
    with open(json_path, "w") as f:
        json.dump(data, f)


def fix_annotations(split_path, metadata_path, split="train"):
    outputPath = split_path.replace(".json", "_fixed.json")
    append_categories(split_path, metadata_path, outputPath)
    swap_categories_ids(outputPath)
