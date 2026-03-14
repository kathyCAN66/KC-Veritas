# calculate entropy, split score
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

def calculate_entropy(labels):
    probabilities = np.bincount(labels) / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small value to avoid log(0)
    return entropy 

def calculate_split_score(df, feature, target):
    total_entropy = calculate_entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = 0
    for value in values:
        subset = df[df[feature] == value]
        subset_entropy = calculate_entropy(subset[target])
        weighted_entropy += (len(subset) / len(df)) * subset_entropy
    split_score = total_entropy - weighted_entropy
    return split_score

def calculate_mutual_information(df, feature, target):
    mi = mutual_info_score(df[feature], df[target])
    return mi

def add_features_to_df(df):
    df['entropy'] = df.apply(lambda row: calculate_entropy(row['scene']), axis=1)
    df['split_score'] = df.apply(lambda row: calculate_split_score(df, 'scene', 'referring_expression'), axis=1)
    df['mutual_information'] = df.apply(lambda row: calculate_mutual_information(df, 'scene', 'referring_expression'), axis=1)
    return df
