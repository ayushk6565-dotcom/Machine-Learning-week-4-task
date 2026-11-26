import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


data = pd.read_csv("Gold Price Prediction/gld_price_data.csv")
df = pd.DataFrame(data)
df = df.drop("Date",axis=1)
X = df.drop("GLD",axis=1)
y = df["GLD"]

Scaler = StandardScaler()
X = Scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8)
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
xgb_model.fit(X_train,y_train)
pred = xgb_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))
