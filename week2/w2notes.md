Best hyperparameters found:
{'criterion': 'entropy', 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 6}
Best cross-validation accuracy: 0.8214
Test set accuracy: 0.8133
              precision    recall  f1-score   support

       color       0.20      0.03      0.05        34
      either       0.86      0.91      0.88       234
         pos       0.64      0.91      0.75        32

    accuracy                           0.81       300
   macro avg       0.57      0.62      0.56       300
weighted avg       0.76      0.81      0.78       300

Notes:
shell commands
pwd
ls
cd
mv
rm

## Git commit steps
1. never commit refer/ or .venv
2. use .gitignore to keep big files / unneeded files from being moved
3. git status | more
4. if accidentally commited smth, use restore, select subfolders to remove (eg .venv/lib/*)
5. need to commit deletions and changes as well as files added
   a. git add, git restore
   b. to commit use git commit -m "commit message"
6. finally do git push

- **Undo mistakes**:
  - Unstage accidentally added files: `git restore --staged <file/folder>`
  - Reset to earlier commit: `git reset --soft|--mixed|--hard <commit_hash>`
- **Local vs remote**:
  - See commits not yet pushed: `git log origin/main..HEAD --oneline`
- **Ignore files/folders**: Add `venv/` to `.gitignore` to prevent tracking virtual environments.

## Python / VS Code
- `os.path.dirname(__file__)` → get folder of current script
- Access files in other folders using `os.path.join(base_dir, "..", "week1", "file.csv")`
- `pathlib.Path(__file__).parent.parent / "week1" / "file.csv"` → cleaner alternative

## Random Forest / Decision Trees
- `RandomForestClassifier` hyperparameters:
  - `n_estimators`, `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `random_state`
- Plotting:
  - `tree.plot_tree()` works only for **DecisionTreeClassifier**, not RandomForest
  - Plot a single tree from forest: `clf.estimators_[0]`
- Feature importance:
  - `clf.feature_importances_` shows most important features across all trees
  - Can plot with matplotlib for better insight
- Hyperparameter tuning for Decision Trees:
  - Use `GridSearchCV` or `RandomizedSearchCV` to automatically search for the best parameters 
  
## Restoring to previous logs
- `git log --oneline` to find hash codes for previous commits
- `git reset --soft [hash code]` of previous safe commit
- if accidentally moved a soft reset:
   - use `git reflog` to find list of hash codes
   - reset to most recent commit
      - `git reset --soft HEAD@{1}`
- to check if everything's ok, use `git status | more`

