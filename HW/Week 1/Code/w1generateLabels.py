import os
import numpy as np
import pandas as pd

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'w1_q_selection_dataset.csv')
df = pd.read_csv(csv_path)

def generate_labels_random(df):
    labels = []
    for _, row in df.iterrows():
        label = np.random.choice(['color', 'pos', 'either'])
        labels.append(label)
    return labels

def generate_labels_split_score(df):
    labels = []
    for _, row in df.iterrows():
        if row['split_score_color'] > row['split_score_pos']:
            labels.append('color')
        elif row['split_score_pos'] > row['split_score_color']:
            labels.append('pos')
        else:
            labels.append('either')
    return labels

def generate_labels_entropy(df):
    labels = []
    for _, row in df.iterrows():
        if row['entropy_dec_color'] > row['entropy_dec_pos']:
            labels.append('color')
        elif row['entropy_dec_pos'] > row['entropy_dec_color']:
            labels.append('pos')
        else:
            labels.append('either')
    return labels

def generate_labels_weighted(df, entropy_weight=0.7, split_weight=0.3):
    labels = []
    for _, row in df.iterrows():
        score_color = entropy_weight * row['entropy_dec_color'] + split_weight * row['split_score_color']
        score_pos = entropy_weight * row['entropy_dec_pos'] + split_weight * row['split_score_pos']
        if score_color > score_pos:
            labels.append('color')
        elif score_pos > score_color:
            labels.append('pos')
        else:
            labels.append('either')
    return labels

# helper function
def _choose_label_from_scores(score_color, score_pos, noise_level=0.0):
    if noise_level > 0:
        score_color += np.random.normal(scale=noise_level)
        score_pos += np.random.normal(scale=noise_level)
    if score_color > score_pos:
        return 'color'
    elif score_pos > score_color:
        return 'pos'
    else:
        return 'either'
    
def generate_labels_split_score_random(df, noise_level=0.1):
    labels = []
    for _, row in df.iterrows():
        label = _choose_label_from_scores(row['split_score_color'], row['split_score_pos'], noise_level)
        labels.append(label)
    return labels

def generate_labels_entropy_random(df, noise_level=0.1):
    labels = []
    for _, row in df.iterrows():
        label = _choose_label_from_scores(row['entropy_dec_color'], row['entropy_dec_pos'], noise_level)
        labels.append(label)
    return labels

def generate_labels_weighted_random(df, entropy_weight=0.7, split_weight=0.3, noise_level=0.1):
    labels = []
    for _, row in df.iterrows():
        sc = entropy_weight * row['entropy_dec_color'] + split_weight * row['split_score_color']
        sp = entropy_weight * row['entropy_dec_pos'] + split_weight * row['split_score_pos']
        label = _choose_label_from_scores(sc, sp, noise_level)
        labels.append(label)
    return labels

if __name__ == '__main__':
    df['labels_random'] = generate_labels_random(df)
    df['labels_split_score'] = generate_labels_split_score(df)
    df['labels_entropy'] = generate_labels_entropy(df)
    df['labels_weighted'] = generate_labels_weighted(df)
    df['labels_entropy_random'] = generate_labels_entropy_random(df)
    df['labels_split_score_random'] = generate_labels_split_score_random(df)
    df['labels_weighted_random'] = generate_labels_weighted_random(df)
    
    out_dir = script_dir
    
    df_random = df[['n', 'entropy', 'split_score_color', 'split_score_pos', 'labels_random']].copy()
    df_random.to_csv(os.path.join(out_dir, 'w1_labels_random.csv'), index=False)
    
    df_split = df[['n', 'entropy', 'split_score_color', 'split_score_pos', 'labels_split_score']].copy()
    df_split.to_csv(os.path.join(out_dir, 'w1_labels_split_score.csv'), index=False)
    
    df_ent = df[['n', 'entropy', 'split_score_color', 'split_score_pos', 'labels_entropy']].copy()
    df_ent.to_csv(os.path.join(out_dir, 'w1_labels_entropy.csv'), index=False)
    
    df_weighted = df[['n', 'entropy', 'split_score_color', 'split_score_pos', 'labels_weighted']].copy()
    df_weighted.to_csv(os.path.join(out_dir, 'w1_labels_weighted.csv'), index=False)
    
    df_ent_rand = df[['n', 'entropy', 'split_score_color', 'split_score_pos', 'labels_entropy_random']].copy()
    df_ent_rand.to_csv(os.path.join(out_dir, 'w1_labels_entropy_random.csv'), index=False)
    
    df_split_rand = df[['n', 'entropy', 'split_score_color', 'split_score_pos', 'labels_split_score_random']].copy()
    df_split_rand.to_csv(os.path.join(out_dir, 'w1_labels_split_score_random.csv'), index=False)
    
    df_weighted_rand = df[['n', 'entropy', 'split_score_color', 'split_score_pos', 'labels_weighted_random']].copy()
    df_weighted_rand.to_csv(os.path.join(out_dir, 'w1_labels_weighted_random.csv'), index=False)
