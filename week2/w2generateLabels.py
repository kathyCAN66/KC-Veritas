import os
import pandas as pd
import numpy as np
import csv
import random

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