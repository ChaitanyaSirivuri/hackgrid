# model_operations.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, top_k_accuracy_score, average_precision_score, brier_score_loss, jaccard_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if len(np.unique(y_test)) > 2:
        average = 'weighted'
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)

        return accuracy, precision, recall, f1, 0, 0, 0

    else:
        average = 'binary'
        roc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        brier = brier_score_loss(y_test, y_pred)
        jaccard = jaccard_score(y_test, y_pred, average=average)

        return accuracy, precision, recall, f1, roc, brier, jaccard


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_models():
    return [
        ("Random Forest", RandomForestClassifier()),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Support Vector Machine", SVC()),
        ("Logistic Regression", LogisticRegression()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Naive Bayes", GaussianNB()),
        ("Gradient Boosting", GradientBoostingClassifier()),
        ("Neural Network", MLPClassifier()),
        ("AdaBoost", AdaBoostClassifier()),
        ("Bagging", BaggingClassifier()),
        ("Extra Trees", ExtraTreesClassifier()),
        ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
        ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
        ("LightGBM", lgb.LGBMClassifier()),
        ("XGBoost", xgb.XGBClassifier())
    ]


def label_encode_data(X, y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = label_encoder.fit_transform(X[column])
    return X, y_encoded


def run_classification_models(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_accuracy = 0.0

    results = []

    for model_name, model in models:
        accuracy, precision, recall, f1, roc, brier, jaccard = evaluate_model(
            model, X_train, y_train, X_test, y_test)
        results.append((model_name, accuracy, precision, recall, f1, roc, brier, jaccard

                        ))

        # Save the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC ROC", "Brier Score", "Jaccard Score"]), best_model


def save_best_model(model, model_name):
    model_filename = f"{model_name}_best_model.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Saved the best model to {model_filename}")
    return model_filename


def start_classification(df, target):
    X = df.drop(target, axis=1)
    y = df[target]

    X, y = label_encode_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    models = create_models()

    results, best_model = run_classification_models(
        models, X_train, y_train, X_test, y_test)

    results = results.sort_values(by='Accuracy', ascending=False)
    results = results.reset_index(drop=True)
    results.index += 1

    model_filename = save_best_model(best_model, "classification")

    return results, model_filename
