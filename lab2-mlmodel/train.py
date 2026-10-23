import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Sample data: advertising spend (thousands) vs sales (units)
data = pd.read_csv("Advertising Budget and Sales.csv")
df = pd.DataFrame(data)

# 2. Input feature and target
X = df[["TV Ad Budget"]]  #
y = df["Sales"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print("Coefficient: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("R2 score: ", r2_score(y_test, y_pred))

# 7. Actual vs predicted
result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred.round(2)})
print(result)