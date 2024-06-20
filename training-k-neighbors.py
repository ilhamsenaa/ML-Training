# Import scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_svc = KNeighborsClassifier()
model_svc.fit(X_train, y_train)

pred_svc = model_svc.predict(X_test)

accuracy_svc = accuracy_score(pred_svc, y_test)
prec_svc = precision_score(pred_svc, y_test, average='macro')
recall_score_svc = recall_score(pred_svc, y_test, average='macro')
f1_score_svc = f1_score(pred_svc, y_test, average='macro')

print(accuracy_svc)
print(prec_svc)
print(recall_score_svc)
print(f1_score_svc)