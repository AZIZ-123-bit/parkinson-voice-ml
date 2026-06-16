import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score,
                             f1_score, recall_score, precision_score,
                             confusion_matrix)

data = pd.read_csv("pd_speech_features.csv", header=1)

print("Number of rows (recordings):", data.shape[0])
print("Number of columns:", data.shape[1])
print("Healthy (0) and Parkinson (1):")
print(data["class"].value_counts())
print()

y = data["class"]
gender = data["gender"]
patient_id = data["id"]
X = data.drop(columns=["id", "class"])

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_index, test_index = next(splitter.split(X, y, groups=patient_id))

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]
y_train = y.iloc[train_index]
y_test = y.iloc[test_index]
gender_test = gender.iloc[test_index]

print("Training recordings:", len(X_train))
print("Testing recordings:", len(X_test))
print()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)

def get_scores(y_true, y_pred, y_proba):
    scores = {}
    scores["Accuracy"] = accuracy_score(y_true, y_pred)
    scores["MCC"] = matthews_corrcoef(y_true, y_pred)
    scores["AUC"] = roc_auc_score(y_true, y_proba)
    scores["F1"] = f1_score(y_true, y_pred)
    scores["Recall"] = recall_score(y_true, y_pred)
    scores["Specificity"] = specificity(y_true, y_pred)
    scores["Precision"] = precision_score(y_true, y_pred)
    return scores

all_results = {}
trained_models = {}

print("Training Logistic Regression ...")
lr = LogisticRegression(max_iter=5000, class_weight="balanced")
lr_grid = GridSearchCV(lr, {"C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=5)
lr_grid.fit(X_train, y_train)

lr_best = lr_grid.best_estimator_
lr_pred = lr_best.predict(X_test)
lr_proba = lr_best.predict_proba(X_test)[:, 1]

all_results["Logistic Regression"] = get_scores(y_test, lr_pred, lr_proba)
trained_models["Logistic Regression"] = lr_best
print("   best C:", lr_grid.best_params_)
print("   AUC =", round(all_results["Logistic Regression"]["AUC"], 3))

print("Training LASSO ...")
lasso = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    max_iter=5000,
    class_weight="balanced"
)

lasso_grid = GridSearchCV(
    lasso,
    {"C": [0.01, 0.05, 0.1, 0.5, 1]},
    scoring="roc_auc",
    cv=5
)

lasso_grid.fit(X_train, y_train)

lasso_best = lasso_grid.best_estimator_
lasso_pred = lasso_best.predict(X_test)
lasso_proba = lasso_best.predict_proba(X_test)[:, 1]

all_results["LASSO"] = get_scores(y_test, lasso_pred, lasso_proba)
trained_models["LASSO"] = lasso_best
print("   best C:", lasso_grid.best_params_)

kept = np.sum(lasso_best.coef_ != 0)
print("   LASSO kept", kept, "features out of", X_train.shape[1])

print("Training Decision Tree ...")
tree = DecisionTreeClassifier(class_weight="balanced", random_state=42)
tree_grid = GridSearchCV(
    tree,
    {"max_depth": [3, 5, 10, None]},
    scoring="roc_auc",
    cv=5
)

tree_grid.fit(X_train, y_train)

tree_best = tree_grid.best_estimator_
tree_pred = tree_best.predict(X_test)
tree_proba = tree_best.predict_proba(X_test)[:, 1]

all_results["Decision Tree"] = get_scores(y_test, tree_pred, tree_proba)
trained_models["Decision Tree"] = tree_best
print("   best depth:", tree_grid.best_params_)

print("Training Random Forest ...")
forest = RandomForestClassifier(class_weight="balanced", random_state=42)

forest_grid = GridSearchCV(
    forest,
    {"n_estimators": [200, 400], "max_depth": [None, 10, 20]},
    scoring="roc_auc",
    cv=5
)

forest_grid.fit(X_train, y_train)

forest_best = forest_grid.best_estimator_
forest_pred = forest_best.predict(X_test)
forest_proba = forest_best.predict_proba(X_test)[:, 1]

all_results["Random Forest"] = get_scores(y_test, forest_pred, forest_proba)
trained_models["Random Forest"] = forest_best
print("   best settings:", forest_grid.best_params_)

print("Training Naive Bayes ...")
bayes = GaussianNB()

bayes_grid = GridSearchCV(
    bayes,
    {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    scoring="roc_auc",
    cv=5
)

bayes_grid.fit(X_train, y_train)

bayes_best = bayes_grid.best_estimator_
bayes_pred = bayes_best.predict(X_test)
bayes_proba = bayes_best.predict_proba(X_test)[:, 1]

all_results["Naive Bayes"] = get_scores(y_test, bayes_pred, bayes_proba)
trained_models["Naive Bayes"] = bayes_best
print("   done")

print("Training kNN ...")
knn = KNeighborsClassifier()

knn_grid = GridSearchCV(
    knn,
    {"n_neighbors": [3, 5, 7, 9, 11, 15]},
    scoring="roc_auc",
    cv=5
)

knn_grid.fit(X_train, y_train)

knn_best = knn_grid.best_estimator_
knn_pred = knn_best.predict(X_test)
knn_proba = knn_best.predict_proba(X_test)[:, 1]

all_results["kNN"] = get_scores(y_test, knn_pred, knn_proba)
trained_models["kNN"] = knn_best
print("   best k:", knn_grid.best_params_)

print("Training Neural Network ...")
nn = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)

nn_grid = GridSearchCV(
    nn,
    {"hidden_layer_sizes": [(64,), (128, 64)]},
    scoring="roc_auc",
    cv=5
)

nn_grid.fit(X_train, y_train)

nn_best = nn_grid.best_estimator_
nn_pred = nn_best.predict(X_test)
nn_proba = nn_best.predict_proba(X_test)[:, 1]

all_results["Neural Network"] = get_scores(y_test, nn_pred, nn_proba)
trained_models["Neural Network"] = nn_best

print("   best settings:", nn_grid.best_params_)
print()

results_table = pd.DataFrame(all_results).T.round(3)
results_table = results_table.sort_values("MCC", ascending=False)

print("=================== MODEL COMPARISON ===================")
print(results_table)

results_table.to_csv("results.csv")

best_name = results_table.index[0]
best_model = trained_models[best_name]

print()
print("Best model:", best_name)

print()
print("=================== FAIRNESS BY GENDER ===================")

best_pred = best_model.predict(X_test)
best_proba = best_model.predict_proba(X_test)[:, 1]

fairness = {}

for g in [0, 1]:
    mask = (gender_test.values == g)

    fairness["gender " + str(g)] = get_scores(
        y_test.values[mask],
        best_pred[mask],
        best_proba[mask]
    )

fairness_table = pd.DataFrame(fairness).T.round(3)

print(fairness_table)

fairness_table.to_csv("fairness.csv")

results_table.plot(kind="bar", figsize=(12, 6))
plt.title("Comparison of models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("comparison.png")

print()
print("Saved: results.csv, fairness.csv, comparison.png")
print("Finished.")