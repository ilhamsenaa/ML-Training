# Import scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

pred_lr = model_lr.predict(X_test)

accuracy_lr = accuracy_score(pred_lr, y_test)
prec_lr = precision_score(pred_lr, y_test, average='macro')
recall_score_lr = recall_score(pred_lr, y_test, average='macro')
f1_score_lr = f1_score(pred_lr, y_test, average='macro')

print(accuracy_lr)
print(prec_lr)
print(recall_score_lr)
print(f1_score_lr)
