import os
import random
import numpy as np
import csv

# seed
random.seed(6)
np.random.seed(6)

def compute_entropy(belief):
    # compute for nonzero values only
    return -np.sum(belief[belief > 0] * np.log2(belief[belief > 0]))

def expected_entropy_decrease(binary_array, belief, n):
    initial_entropy = compute_entropy(belief)
    
    ones = sum(1 for bit in binary_array if bit == '1')
    zeros = n - ones
    
    if ones == 0 or zeros == 0:
        return 0 # no entropy decrease
    
    p_ones = ones / n
    p_zeros = zeros / n
    
    belief_ones = belief * (binary_array == '1')
    belief_zeros = belief * (binary_array == '0')
    
    entropy_ones = compute_entropy(belief_ones)
    entropy_zeros = compute_entropy(belief_zeros)
    
    expected_final_entropy = p_ones * entropy_ones + p_zeros * entropy_zeros
    
    return initial_entropy - expected_final_entropy

def split_score(binary_array, n):
    # split score formula: -((# of ones/n)*log_2(# of ones/n) + (# of zeros/n)*log_2(# of zeros/n))
    ones = sum(1 for bit in binary_array if bit == '1')
    zeros = n - ones
    
    split_score = 0
    if ones > 0:
        split_score -= (ones / n) * np.log2(ones / n)
    if zeros > 0:
        split_score -= (zeros / n) * np.log2(zeros / n)
    
    return split_score

def generate_datapoint():
    n = random.randint(2, 6)
    
    belief_not_rounded = np.random.dirichlet(np.ones(n))
    belief = np.round(belief_not_rounded, 2)
    belief[-1] = round(1 - np.sum(belief[:-1]), 2)
    
    color = ''.join(random.choice('01') for _ in range(n))
    pos = ''.join(random.choice('01') for _ in range(n))

    entropy = compute_entropy(belief)

    split_score_color = split_score(color, n)
    split_score_pos = split_score(pos, n)

    entropy_dec_color = expected_entropy_decrease(color, belief, n)
    entropy_dec_pos = expected_entropy_decrease(pos, belief, n)

    return [n, belief.tolist(), color, pos, entropy, split_score_color, split_score_pos, entropy_dec_color, entropy_dec_pos]

def generate_dataset(num_samples, filename=None):
    seen_beliefs = set()
    datapoints = []

    while len(datapoints) < num_samples:
        datapoint = generate_datapoint()
        belief_tuple = tuple(datapoint[1])
        
        if belief_tuple not in seen_beliefs:
            seen_beliefs.add(belief_tuple)
            datapoints.append(datapoint)

    if filename is None:
        # write file to same directory as this script
        script_dir = os.path.dirname(__file__)
        filename = os.path.join(script_dir, 'w1_q_selection_dataset.csv')

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n', 'belief', 'color', 'pos', 'entropy', 'split_score_color', 'split_score_pos', 'entropy_dec_color', 'entropy_dec_pos'])
        for datapoint in datapoints:
            writer.writerow(datapoint)


if __name__ == "__main__":   
    generate_dataset(1000)

