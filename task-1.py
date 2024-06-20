from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import numpy as np

california = datasets.fetch_california_housing()
X = california.data
y = california.target

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
