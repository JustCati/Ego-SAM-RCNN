import json


def append_categories(path_1, path_2, outPath):
    with open(path_1, "r") as f:
        eval = json.load(f)
    with open(path_2, "r") as f:
        metadata = json.load(f)

    eval["categories"] = metadata["categories"]
    with open(outPath, "w") as f:
        json.dump(eval, f)


