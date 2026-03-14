import os
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# load datasets
script_dir = os.path.dirname(__file__)

dtype_spec = {'color': str, 'pos': str}
# df_random = pd.read_csv(os.path.join(script_dir, 'w1_labels_random.csv'), dtype=dtype_spec)
df_entropy = pd.read_csv(os.path.join(script_dir, 'w1_labels_entropy.csv'), dtype=dtype_spec)
# df_split = pd.read_csv(os.path.join(script_dir, 'w1_labels_split_score.csv'), dtype=dtype_spec)
# df_weighted = pd.read_csv(os.path.join(script_dir, 'w1_labels_weighted.csv'), dtype=dtype_spec)
# df_entropy_random = pd.read_csv(os.path.join(script_dir, 'w1_labels_entropy_random.csv'), dtype=dtype_spec)
# df_split_random = pd.read_csv(os.path.join(script_dir, 'w1_labels_split_score_random.csv'), dtype=dtype_spec)
# df_weighted_random = pd.read_csv(os.path.join(script_dir, 'w1_labels_weighted_random.csv'), dtype=dtype_spec)

# train/test split
def split_dataset(df):
    X = df.iloc[:, 0:3]
    y = df.iloc[:, 4]
    return train_test_split(X, y, test_size=0.3, random_state=6)

# train using gini index
def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=6, max_depth=2, min_samples_leaf=2)
    clf_gini.fit(X_train, y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=6, max_depth=2, min_samples_leaf=2)
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def evaluation(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(f'Results:\n{y_pred[:10]}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

if __name__ == "__main__":
    # for df, name in [(df_random, 'Random'), (df_entropy, 'Entropy'), (df_split, 'Split Score'), (df_weighted, 'Weighted'), (df_entropy_random, 'Entropy Random'), (df_split_random, 'Split Score Random'), (df_weighted_random, 'Weighted Random')]:
    for df, name in [(df_entropy, 'Entropy')]:
        print(f'\nTraining on {name} dataset:')
        print(f'\nTraining on {name} dataset:')
        X_train, X_test, y_train, y_test = split_dataset(df)
        clf_gini = train_using_gini(X_train, y_train)
        print('Gini Index:')
        evaluation(clf_gini, X_test, y_test)
        clf_entropy = train_using_entropy(X_train, y_train)
        print('Entropy:')
        evaluation(clf_entropy, X_test, y_test)
        tree.plot_tree(clf_gini)
        plt.title(f'{name} - Gini Decision Tree')
        plt.show()
        tree.plot_tree(clf_entropy)
        plt.title(f'{name} - Entropy Decision Tree')
        plt.show()