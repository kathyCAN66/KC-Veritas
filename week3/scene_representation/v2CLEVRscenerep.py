import json
import csv
from collections import Counter

with open("/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/CLEVR_train_scenes.json", "r") as f:
    data = json.load(f)

scenes = data["scenes"]

rows = []

for scene in scenes:
    objects = scene["objects"]
    relationships = scene["relationships"]

    color_hist = Counter()
    shape_hist = Counter()
    size_hist = Counter()
    texture_hist = Counter()

    position_lr = Counter({"left": 0, "right": 0})
    position_fb = Counter({"front": 0, "back": 0})

    for i, obj in enumerate(objects):
        # obj attributes
        color_hist[obj["color"]] += 1
        shape_hist[obj["shape"]] += 1
        size_hist[obj["size"]] += 1
        texture_hist[obj["material"]] += 1

        # positional indices
        r_to_l = len(relationships["right"][i])
        f_to_b = len(relationships["front"][i])

        n = len(objects)

        # left/right
        if r_to_l >= n // 2:
            position_lr["left"] += 1
        else:
            position_lr["right"] += 1

        # front/back
        if f_to_b >= n // 2:
            position_fb["back"] += 1
        else:
            position_fb["front"] += 1

    scene_hist = {
        "color": dict(color_hist),
        "shape": dict(shape_hist),
        "size": dict(size_hist),
        "texture": dict(texture_hist),
        "RtoL_position": dict(position_lr),
        "FtoB_position": dict(position_fb)
    }

    rows.append({
        "image_index": scene["image_index"],
        "histogram": scene_hist
    })

# CSV
with open("/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/hist_CLEVR_train_scenes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_index", "histogram"])

    for row in rows:
        writer.writerow([row["image_index"], str(row["histogram"])])

print("Histogram CSV created")