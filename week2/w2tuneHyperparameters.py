import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
base_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_dir, "..", "week1", "w1_labels_entropy.csv"))

X = df.iloc[:, 0:3]
y = df.iloc[:, 4]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

# Model
clf = DecisionTreeClassifier(random_state=6)

# Hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 12, 15],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

# Grid search
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# Fit
grid_search.fit(X_train, y_train)

# Best hyperparamters and score
print("Best hyperparameters found:")
print(grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))