import csv
import ast
import numpy as np
import random
from collections import Counter, defaultdict

input_path = "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/2k_dict_CLEVR_scenes_ambiguous.csv"
output_path = "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/final_training_dataset.csv"

ATTRS = ["color", "size", "shape", "texture", "posLR", "posFB"]

# helper fuctions

def compute_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def build_groups(objects, attr):
    groups = defaultdict(list)

    if attr in ["color", "size", "shape", "texture"]:
        for obj in objects:
            groups[obj[attr]].append(obj)

    elif attr == "posLR":
        n = len(objects)
        for obj in objects:
            if obj["RtoL"] >= n // 2:
                groups["left"].append(obj)
            else:
                groups["right"].append(obj)

    elif attr == "posFB":
        n = len(objects)
        for obj in objects:
            if obj["FtoB"] >= n // 2:
                groups["back"].append(obj)
            else:
                groups["front"].append(obj)

    return groups


def expected_entropy_decrease(objects, attr):
    n = len(objects)
    if n <= 1:
        return 0

    belief = np.ones(n) / n
    initial_entropy = compute_entropy(belief)

    groups = build_groups(objects, attr)

    expected_entropy = 0

    for group in groups.values():
        p = len(group) / n
        if p == 0:
            continue
        group_belief = np.ones(len(group)) / len(group)
        expected_entropy += p * compute_entropy(group_belief)

    return initial_entropy - expected_entropy


def split_score(objects, attr):
    n = len(objects)
    groups = build_groups(objects, attr)

    score = 0
    for group in groups.values():
        p = len(group) / n
        if p > 0:
            score -= p * np.log2(p)
    return score


def filter_by_ref(objects, ref_expr):
    key = list(ref_expr.keys())[0]
    val = ref_expr[key]
    return [obj for obj in objects if obj[key] == val]


def elimination_score(objects, attr, gt_obj):
    groups = build_groups(objects, attr)

    # find group containing GT
    for group in groups.values():
        if gt_obj in group:
            return len(group)
    return len(objects)


# main

results = []

with open(input_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        image_index = row[0]
        objects = ast.literal_eval(row[1])
        gt_object = ast.literal_eval(row[2])
        ref_expr = ast.literal_eval(row[3])

        candidates = filter_by_ref(objects, ref_expr)
        n = len(candidates)

        if n <= 1:
            continue

        # Compute splits + gains
        splits = {}
        gains = {}

        for attr in ATTRS:
            splits[attr] = split_score(candidates, attr)
            gains[attr] = expected_entropy_decrease(candidates, attr)

        # gain_label
        max_gain = max(gains.values())
        best_gain = [k for k, v in gains.items() if v == max_gain]

        gain_label = best_gain[0] if len(best_gain) == 1 else "ask_any"

        # elim_label
        remaining_attrs = [a for a in ATTRS if a not in ref_expr]

        elim_scores = {}
        for attr in remaining_attrs:
            elim_scores[attr] = elimination_score(candidates, attr, gt_object)

        min_elim = min(elim_scores.values())
        best_elim = [k for k, v in elim_scores.items() if v == min_elim]

        elim_label = best_elim[0] if len(best_elim) == 1 else "ask_any"

        results.append([
            n,
            splits["color"],
            splits["size"],
            splits["texture"],
            splits["shape"],
            splits["posLR"],
            splits["posFB"],
            gains["color"],
            gains["size"],
            gains["texture"],
            gains["shape"],
            gains["posLR"],
            gains["posFB"],
            gain_label,
            elim_label
        ])


columns = [
    "num_objects",
    "color_split",
    "size_split",
    "texture_split",
    "shape_split",
    "posLR_split",
    "posFB_split",
    "color_gain",
    "size_gain",
    "texture_gain",
    "shape_gain",
    "posLR_gain",
    "posFB_gain",
    "gain_label",
    "elim_label"
]

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(results)

print("Final training dataset created")