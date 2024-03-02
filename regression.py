import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, d2_pinball_score, d2_tweedie_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor
import xgboost as xgb
import lightgbm as lgb


def remove_nan_values(X):
    return X.dropna()


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    d2_pinball = d2_pinball_score(y_test, y_pred)
    d2_tweedie = d2_tweedie_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return mae, mse, rmse, r2, d2_pinball, d2_tweedie, mape


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_models():
    return [
        ("Random Forest", RandomForestRegressor()),
        ("Gradient Boosting", GradientBoostingRegressor()),
        ("AdaBoost", AdaBoostRegressor()),
        ("Extra Trees", ExtraTreesRegressor()),
        ("Bagging", BaggingRegressor()),
        ("HistGradientBoosting", HistGradientBoostingRegressor()),
        ("Isolation Forest", IsolationForest()),
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge()),
        ("Lasso", Lasso()),
        ("Elastic Net", ElasticNet()),
        ("Bayesian Ridge", BayesianRidge()),
        ("Huber Regressor", HuberRegressor()),
        ("SGD Regressor", SGDRegressor()),
        ("LightGBM", lgb.LGBMRegressor()),
        ("XGBoost", xgb.XGBRegressor())
    ]


def run_regression_models(models, X_train, y_train, X_test, y_test):
    results = []
    best_model = None
    best_mse = float('inf')  # Initialize with a high value
    for name, model in models:
        try:
            score = evaluate_model(model, X_train, y_train, X_test, y_test)
            results.append({"Model": name, "MAE": score[0], "MSE": score[1], "RMSE": score[2],
                           "R2": score[3], "D2_Pinball": score[4], "D2_Tweedie": score[5], "MAPE": score[6]})
            if score[0] < best_mse:
                best_mse = score[0]
                best_model = model
        except Exception as e:
            results.append({"Model": name, "Error": str(e)})

    return results, best_model


def save_best_model(model, model_filename):
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    return model_filename


def start_regression(df, target):
    df = remove_nan_values(df)
    X = df.drop(target, axis=1)
    y = df[target]

    # Remove NaN values
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    results, best_model = run_regression_models(
        create_models(), X_train, y_train, X_test, y_test)

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Sort the DataFrame by MSE in ascending order
    results_df = results_df.sort_values(by='MSE')
    results_df = results_df.reset_index(drop=True)
    results_df.index += 1

    # Save the sorted DataFrame to a CSV file (optional)
    results_df.to_csv("sorted_results.csv", index=False)

    model_filename = save_best_model(best_model, "best_model.pkl")
    return results_df, model_filename
