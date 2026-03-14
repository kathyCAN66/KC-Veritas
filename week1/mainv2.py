import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

script_dir = os.path.dirname(__file__)
dtype_spec = {'color': str, 'pos': str}

datasets = {
    'Random': 'w1_labels_random.csv',
    'Entropy': 'w1_labels_entropy.csv',
    'SplitScore': 'w1_labels_split_score.csv',
    'Weighted': 'w1_labels_weighted.csv',
    'EntropyRandom': 'w1_labels_entropy_random.csv',
    'SplitScoreRandom': 'w1_labels_split_score_random.csv',
    'WeightedRandom': 'w1_labels_weighted_random.csv'
}


def split_dataset(df):
    X = df.iloc[:, 0:4]
    y = df.iloc[:, 4]
    return train_test_split(X, y, test_size=0.3, random_state=6)

def train_decision_tree(X_train, y_train, criterion='gini'):
    clf = DecisionTreeClassifier(criterion=criterion, random_state=6, max_depth=3, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    return clf

def export_results(X_test, y_test, y_pred, output_path):
    df_export = X_test.copy()
    df_export['predicted'] = y_pred
    df_export['real'] = y_test.values
    df_export.to_csv(output_path, index=False)
    print(f'Exported results to {output_path}')


if __name__ == "__main__":
    for name, file in datasets.items():
        print(f'\nProcessing {name} dataset...')
        df = pd.read_csv(os.path.join(script_dir, file), dtype=dtype_spec)

        # Split
        X_train, X_test, y_train, y_test = split_dataset(df)

        # Train Gini
        clf_gini = train_decision_tree(X_train, y_train, criterion='gini')
        y_pred_gini = clf_gini.predict(X_test)

        # Export
        output_gini = os.path.join(script_dir, f'{name}_Gini_results.csv')
        export_results(X_test, y_test, y_pred_gini, output_gini)

        # Train Entropy
        clf_entropy = train_decision_tree(X_train, y_train, criterion='entropy')
        y_pred_entropy = clf_entropy.predict(X_test)

        # Export
        output_entropy = os.path.join(script_dir, f'{name}_Entropy_results.csv')
        export_results(X_test, y_test, y_pred_entropy, output_entropy)