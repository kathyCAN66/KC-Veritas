import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# dataset
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/CLEVR_training_dataset.csv")
df = pd.read_csv(file_path)

# features
FEATURE_COLS = [
    "num_objects",
    "color_split",
    "size_split",
    "texture_split",
    "shape_split",
    "posLR_split",
    "posFB_split",
    "color_gain",
    "size_gain",
    "texture_gain",
    "shape_gain",
    "posLR_gain",
    "posFB_gain"
]

# split function
def dataset_split(df, label_col):
    X = df[FEATURE_COLS]
    y = df[label_col]

    return train_test_split(
        X, y,
        test_size=0.3,
        random_state=6,
        stratify=y  # keeps label distribution balanced
    )

# train
def train(X_train, y_train,
          max_depth=20,
          n_estimators=200,
          min_samples_split=2,
          min_samples_leaf=4):

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

# evaluation
def evaluation(model, X_test, y_test, label_name):
    y_pred = model.predict(X_test)

    print(f"\n--- {label_name} MODEL ---")
    print(f'Results (first 10 predictions):\n{y_pred[:10]}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# main
if __name__ == "__main__":

    print("\nTraining models on CLEVR-derived dataset")

    # model 1 : gain_label
    X_train, X_test, y_train, y_test = dataset_split(df, "gain_label")
    gain_model = train(X_train, y_train)
    evaluation(gain_model, X_test, y_test, "GAIN LABEL")

    # model 2 : elim_label
    X_train, X_test, y_train, y_test = dataset_split(df, "elim_label")
    elim_model = train(X_train, y_train)
    evaluation(elim_model, X_test, y_test, "ELIM LABEL")