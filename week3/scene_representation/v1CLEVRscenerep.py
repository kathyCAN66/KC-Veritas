import json
import csv

# Load scenes
with open("/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/CLEVR_train_scenes.json", "r") as f:
    data = json.load(f)

scenes = data["scenes"]

rows = []

for scene in scenes:
    objects = scene["objects"]
    relationships = scene["relationships"]

    obj_list = []

    for i, obj in enumerate(objects):
        # from "relationships"
        r_to_l = len(relationships["right"][i])
        f_to_b = len(relationships["front"][i])

        obj_dict = {
            "index": i,
            "shape": obj["shape"],
            "size": obj["size"],
            "color": obj["color"],
            "texture": obj["material"],
            "RtoL": r_to_l,
            "FtoB": f_to_b
        }

        obj_list.append(obj_dict)

    rows.append({
        "image_index": scene["image_index"],
        "objects": obj_list
    })

# CSV
with open("v1_CLEVR_train_scenes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_index", "objects"])

    for row in rows:
        writer.writerow([row["image_index"], str(row["objects"])])

print("CSV created: v1_CLEVR_train_scenes.csv")