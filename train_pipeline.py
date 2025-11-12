import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris, load_boston
import joblib, os, sys

def train_model(model_type="random_forest"):
    if model_type in ["random_forest", "decision_tree"]:
        data = load_iris(as_frame=True)
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42) if model_type=="random_forest" else DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, f"{model_type}_model.pkl")
        print(f"{model_type} accuracy:", acc)
        return acc

    elif model_type == "linear_regression":
        data = load_boston()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        joblib.dump(model, "linear_regression_model.pkl")
        print("linear_regression MSE:", mse)
        return mse

if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    os.chdir("artifacts")
    metric = train_model(sys.argv[1] if len(sys.argv) > 1 else "random_forest")
    print("Training complete. Metric:", metric)
