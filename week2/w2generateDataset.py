import pandas as pd
import ast
import numpy as np
import random

# entropy
def compute_entropy(belief):
    return -np.sum(belief[belief > 0] * np.log2(belief[belief > 0]))

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

    # normalize (important!)
    if belief_ones.sum() > 0:
        belief_ones = belief_ones / belief_ones.sum()
    if belief_zeros.sum() > 0:
        belief_zeros = belief_zeros / belief_zeros.sum()

    entropy_ones = compute_entropy(belief_ones)
    entropy_zeros = compute_entropy(belief_zeros)

    expected_final_entropy = p_ones * entropy_ones + p_zeros * entropy_zeros

    return initial_entropy - expected_final_entropy


# split score
def split_score(binary_array, n):
    ones = sum(1 for bit in binary_array if bit == '1')
    zeros = n - ones

    score = 0
    if ones > 0:
        score -= (ones / n) * np.log2(ones / n)
    if zeros > 0:
        score -= (zeros / n) * np.log2(zeros / n)

    return score

def filter_by_ref(objects, ref_expr):
    return [obj for obj in objects if obj['type'] == ref_expr]


def build_binary_array(objects, attr):
    if attr == "color":
        return ['1' if obj['color'] == 'red' else '0' for obj in objects]

    elif attr == "size":
        return ['1' if obj['size'] == 'big' else '0' for obj in objects]

    elif attr == "position":
        n = len(objects)
        binary = []
        for i in range(n):
            if i < n // 2:
                binary.append('1')   # left
            else:
                binary.append('0')   # right (center included)
        return binary


df = pd.read_csv("/Users/kathychen/PycharmProjects/KC-Veritas/week2/generated_scenes.csv")

results = []

for _, row in df.iterrows():
    objects = ast.literal_eval(row["scene"])
    ref_expr = row["referring_expression"]

    total_objects = len(objects)
    num_cups = sum(1 for obj in objects if obj['type'] == 'cup')
    num_bottles = sum(1 for obj in objects if obj['type'] == 'bottle')

    candidates = filter_by_ref(objects, ref_expr)
    n = len(candidates)

    if n == 0:
        continue

    # uniform belief over candidates
    belief = np.ones(n) / n

    # build splits
    color_bin = build_binary_array(candidates, "color")
    size_bin = build_binary_array(candidates, "size")
    pos_bin = build_binary_array(candidates, "position")

    # split scores
    color_split = split_score(color_bin, n)
    size_split = split_score(size_bin, n)
    pos_split = split_score(pos_bin, n)

    # entropy decrease
    color_gain = expected_entropy_decrease(color_bin, belief, n)
    size_gain = expected_entropy_decrease(size_bin, belief, n)
    pos_gain = expected_entropy_decrease(pos_bin, belief, n)

    gains = {
        "color": color_gain,
        "size": size_gain,
        "position": pos_gain
    }

    max_gain = max(gains.values())
    best_options = [k for k, v in gains.items() if v == max_gain]
    label = random.choice(best_options)

    results.append([
        total_objects,
        num_cups,
        num_bottles,
        color_split,
        size_split,
        pos_split,
        color_gain,
        size_gain,
        pos_gain,
        label
    ])

columns = [
    "num_objects",
    "num_cups",
    "num_bottles",
    "split_score_color",
    "split_score_size",
    "split_score_position",
    "entropy_dec_color",
    "entropy_dec_size",
    "entropy_dec_position",
    "label"
]

out_df = pd.DataFrame(results, columns=columns)
out_df.to_csv("labeled_scenes.csv", index=False)

print(out_df.head())