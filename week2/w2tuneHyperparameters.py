import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "labeled_scenes.csv")
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
    return train_test_split(X, y, test_size=0.3, random_state=6, stratify=y)

X_train, X_test, y_train, y_test = split_dataset(df)

clf = DecisionTreeClassifier(random_state=6)

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 12, 15],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}


grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best hyperparameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))