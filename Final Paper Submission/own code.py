import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
import numpy as np

# SCIKIT random blobs generation
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=3, n_features=2,random_state=0)

#plotting
gmm = GMM(n_components=5).fit(X)
labels = gmm.predict(X)
color=['blue','green','pink', 'yellow','purple']
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='plasma');
plt.show()