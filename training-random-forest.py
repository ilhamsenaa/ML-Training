# Import scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

pred_rf = model_rf.predict(X_test)

accuracy_rf = accuracy_score(pred_rf, y_test)
prec_rf = precision_score(pred_rf, y_test, average='macro')
recall_score_rf = recall_score(pred_rf, y_test, average='macro')
f1_score_rf = f1_score(pred_rf, y_test, average='macro')

print(accuracy_rf)
print(prec_rf)
print(recall_score_rf)
print(f1_score_rf)