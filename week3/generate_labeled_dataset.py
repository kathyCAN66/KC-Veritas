import csv
import ast
import random
from collections import defaultdict

input_path = "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/2k_dict_CLEVR_train_scenes.csv"
output_path = "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/2k_dict_CLEVR_scenes_ambiguous.csv"

ATTRS = ["color", "size", "shape", "texture"]

rows_out = []

with open(input_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        image_index = row[0]
        objects = ast.literal_eval(row[1])

        # Build attribute → value → list of objects
        valid_choices = []

        for attr in ATTRS:
            value_map = defaultdict(list)
            for obj in objects:
                value_map[obj[attr]].append(obj)

            # Keep only values with >= 2 objects (ambiguous)
            for val, objs in value_map.items():
                if len(objs) >= 2:
                    valid_choices.append((attr, val, objs))

        # Skip scenes with no ambiguity (rare but possible)
        if not valid_choices:
            continue

        # Pick one ambiguous attribute
        attr, val, candidates = random.choice(valid_choices)

        # Pick ground truth object from candidates
        gt_object = random.choice(candidates)

        referring_expression = {attr: val}

        rows_out.append([
            image_index,
            str(objects),
            str(gt_object),
            str(referring_expression)
        ])

# Write CSV
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_index", "objects", "ground_truth", "referring_expression"])
    writer.writerows(rows_out)

print("Ambiguous dataset created")