# Import scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

pred_dt = model_dt.predict(X_test)

accuracy_dt = accuracy_score(pred_dt, y_test)
prec_dt = precision_score(pred_dt, y_test, average='macro')
recall_score_dt = recall_score(pred_dt, y_test, average='macro')
f1_score_dt = f1_score(pred_dt, y_test, average='macro')

print(accuracy_dt)
print(prec_dt)
print(recall_score_dt)
print(f1_score_dt)