import json


def append_categories(path_1, path_2, outPath):
    with open(path_1, "r") as f:
        eval = json.load(f)
    with open(path_2, "r") as f:
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
