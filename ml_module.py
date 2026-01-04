from data_loader import load_los
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
def run_los_regression():
    df = load_los()
    target_col = "lengthofstay"
    y = df[target_col]
    X = df.select_dtypes(include=["int64", "float64"]).drop(columns=[target_col], errors="ignore")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}, Target range: {y.min():.1f}-{y.max():.1f} days")
    X = X.fillna(0)
    y = y.fillna(y.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"MAE : {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R²  : {r2:.3f}")
    # SAVE MODEL
    joblib.dump(model, "models/rf_los.pkl")
    print("Saved RandomForest model to models/rf_los.pkl")
    print("Classical ML module ✅ COMPLETE")
if __name__ == "__main__":
    run_los_regression()
