from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.stats import mode
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# Map cluster labels to actual labels
labels = np.zeros_like(y_pred)
for i in range(3):
    mask = (y_pred == i)
    labels[mask] = mode(y[mask])[0]

# Evaluate the clustering performance
accuracy = np.sum(labels == y) / len(y)
print(f'Accuracy: {accuracy}')