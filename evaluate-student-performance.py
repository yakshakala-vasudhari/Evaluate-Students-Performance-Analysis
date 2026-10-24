from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = Path(__file__).parent / "student-mat.csv"
df = pd.read_csv(path, sep=";")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(path, sep=";")
X = df[["studytime", "absences", "failures"]]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")


evaluate_model("Linear Regression", y_test, predictions)

evaluate_model(
    "Mean Baseline",
    y_test,
    baseline_predictions
)

residuals = y_test - predictions

print(residuals.describe())

plt.figure(figsize=(7, 5))

plt.scatter(y_test, predictions)

plt.xlabel("Actual final grade")
plt.ylabel("Predicted final grade")
plt.title("Actual vs Predicted Grades")

plt.tight_layout()
plt.show()

error_analysis = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions
})

error_analysis["residual"] = (
    error_analysis["actual"]
    - error_analysis["predicted"]
)

error_analysis["absolute_error"] = (
    error_analysis["residual"].abs()
)

largest_errors = error_analysis.sort_values(
    "absolute_error",
    ascending=False
).head(5)

print(largest_errors.round(2))


# Compare an expanded feature set using G1 and G2

X_extended = df[
    ["G1", "G2", "studytime", "absences", "failures"]
]

X_train_ext, X_test_ext, y_train_ext, y_test_ext = (
    train_test_split(
        X_extended,
        y,
        test_size=0.2,
        random_state=42
    )
)

extended_model = LinearRegression()

extended_model.fit(X_train_ext, y_train_ext)

extended_predictions = extended_model.predict(X_test_ext)

evaluate_model(
    "Linear Regression with G1 and G2",
    y_test_ext,
    extended_predictions
)