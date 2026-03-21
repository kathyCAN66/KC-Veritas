# random forest classifier
import os
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "labeled_scenes.csv")  # your labeled scenes file
df = pd.read_csv(file_path)

def split_dataset(df):
    feature_cols = [
        "num_objects",
        "num_cups",
        "num_bottles",
        "split_score_color",
        "split_score_size",
        "split_score_position",
        "entropy_dec_color",
        "entropy_dec_size",
        "entropy_dec_position"
    ]
    X = df[feature_cols]
    y = df["label"]
    return train_test_split(X, y, test_size=0.3, random_state=6)

def train_using_gini(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=6,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def train_using_entropy(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=200,
        criterion="entropy",
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=6,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def evaluation(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(f'Results (first 10 predictions):\n{y_pred[:10]}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

if __name__ == "__main__":
    print(f'\nTraining on Labeled Scenes dataset:')
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Gini
    clf_gini = train_using_gini(X_train, y_train)
    print('\nGini Index:')
    evaluation(clf_gini, X_test, y_test)

    # Entropy
    clf_entropy = train_using_entropy(X_train, y_train)
    print('\nEntropy:')
    evaluation(clf_entropy, X_test, y_test)

    # Plot first tree from Random Forest (Gini)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        clf_gini.estimators_[0],
        feature_names=X_train.columns,
        class_names=clf_gini.classes_,
        filled=True
    )
    plt.title('Labeled Scenes - Gini Tree (from Random Forest)')
    plt.show()

    # Plot first tree from Random Forest (Entropy)
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        clf_entropy.estimators_[0],
        feature_names=X_train.columns,
        class_names=clf_entropy.classes_,
        filled=True
    )
    plt.title('Labeled Scenes - Entropy Tree (from Random Forest)')
    plt.show()