# Import scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
# Load data
california = datasets.fetch_california_housing()

X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# model_logistic = LogisticRegression()
# model_logistic.fit(X_train, y_train)

y_pred = model.predict(X_test)
# y_logistic_pred = model_logistic.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print(y_pred)
print(y_test)
# print(california.data)