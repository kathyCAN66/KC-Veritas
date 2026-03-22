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
file_path = os.path.join(base_dir, "labeled_scenes.csv")
df = pd.read_csv(file_path)

def dataset_split(df):
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

def train(X_train, y_train, max_depth=20, n_estimators=200, min_samples_split=2, min_samples_leaf=4):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        random_state=6,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Results (first 10 predictions):\n{y_pred[:10]}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

if __name__ == "__main__":
    print(f'\nTraining on Labeled Scenes dataset:')
    X_train, X_test, y_train, y_test = dataset_split(df)

    model = train(X_train, y_train)
    evaluation(model, X_test, y_test)