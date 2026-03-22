import pandas as pd
import numpy as np
import ast
import random

# elim incorrect objects in array
def filter_by_ref(objects, ref_expr):
    return [obj for obj in objects if obj['type'] == ref_expr]


# create a binary array for all atributes 
def build_binary_array(objects, attr):
    if attr == "color":
        return np.array(['1' if obj['color'] == 'red' else '0' for obj in objects])

    elif attr == "size":
        return np.array(['1' if obj['size'] == 'big' else '0' for obj in objects])

    elif attr == "position":
        n = len(objects)
        binary = []
        for i in range(n):
            if i < n // 2:
                binary.append('1')   # left
            else:
                binary.append('0')   # right + center
        return np.array(binary)


# make beliefs for correct objects sum to 1
def normalize_belief(belief):
    total = np.sum(belief)
    if total == 0:
        return belief
    belief = belief / total
    belief = np.round(belief, 3)
    belief[-1] = round(1 - np.sum(belief[:-1]), 3)
    return belief


# compute split score
def split_score(binary_array, n):
    ones = np.sum(binary_array == '1')
    zeros = n - ones

    score = 0
    if ones > 0:
        score -= (ones / n) * np.log2(ones / n)
    if zeros > 0:
        score -= (zeros / n) * np.log2(zeros / n)

    return score


# compute expected entropy dec
def compute_entropy(belief):
    belief = belief[belief > 0]
    return -np.sum(belief * np.log2(belief))


def expected_entropy_decrease(binary_array, belief, n):
    binary_array = np.array(binary_array)

    initial_entropy = compute_entropy(belief)

    ones = np.sum(binary_array == '1')
    zeros = n - ones

    if ones == 0 or zeros == 0:
        return 0

    p_ones = ones / n
    p_zeros = zeros / n

    belief_ones = belief * (binary_array == '1')
    belief_zeros = belief * (binary_array == '0')

    # normalize after split
    belief_ones = normalize_belief(belief_ones)
    belief_zeros = normalize_belief(belief_zeros)

    entropy_ones = compute_entropy(belief_ones)
    entropy_zeros = compute_entropy(belief_zeros)

    expected_final_entropy = p_ones * entropy_ones + p_zeros * entropy_zeros

    return initial_entropy - expected_final_entropy


# label
def choose_label(color_gain, size_gain, pos_gain):
    gains = {
        "ask_color": color_gain,
        "ask_size": size_gain,
        "ask_position": pos_gain
    }

    max_gain = max(gains.values())
    best = [k for k, v in gains.items() if v == max_gain]

    if len(best) == 1:
        return best[0]
    else:
        return "ask_any"    # if tie


# compile features for scene
def extract_features(objects, ref_expr):
    objects = list(objects)

    total_objects = len(objects)
    num_cups = sum(1 for obj in objects if obj['type'] == 'cup')
    num_bottles = sum(1 for obj in objects if obj['type'] == 'bottle')

    # belief for all objects
    belief = np.array([obj['belief'] for obj in objects])

    n = len(objects)

    # binary splits
    color_bin = build_binary_array(objects, "color")
    size_bin = build_binary_array(objects, "size")
    pos_bin = build_binary_array(objects, "position")

    # split scores
    color_split = split_score(color_bin, n)
    size_split = split_score(size_bin, n)
    pos_split = split_score(pos_bin, n)

    # entropy decrease
    color_gain = expected_entropy_decrease(color_bin, belief, n)
    size_gain = expected_entropy_decrease(size_bin, belief, n)
    pos_gain = expected_entropy_decrease(pos_bin, belief, n)

    # label
    label = choose_label(color_gain, size_gain, pos_gain)

    return [
        total_objects,
        num_cups,
        num_bottles,
        color_gain,
        pos_gain,
        size_gain,
        color_split,
        pos_split,
        size_split,
        label
    ]

df = pd.read_csv("generated_scenes.csv")

rows = []

for _, row in df.iterrows():
    objects = ast.literal_eval(row["scene"])
    ref_expr = row["referring_expression"]

    features = extract_features(objects, ref_expr)
    rows.append(features)

columns = [
    "num_objects",
    "num_cups",
    "num_bottles",
    "entropy_dec_color",
    "entropy_dec_position",
    "entropy_dec_size",
    "split_score_color",
    "split_score_position",
    "split_score_size",
    "label"
]

out_df = pd.DataFrame(rows, columns=columns)
out_df.to_csv("labeled_scenes.csv", index=False)

print(out_df.head())