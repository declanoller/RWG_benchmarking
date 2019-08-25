from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


N = 500
mu = [5, 4]
theta = np.radians(-30)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c,-s), (s, c)))
gnr = np.random.randn(N, 2)*[3, 1]
print(gnr.shape)
plt.plot(*gnr.transpose(), 'o')
gnr = np.dot(gnr, R) + mu
plt.plot(*gnr.transpose(), 'o')

pca = PCA(n_components=2)
pca.fit(gnr)
print('svd: ', pca.singular_values_)
print('comps: ', pca.components_)
print('mean: ', pca.mean_)

plt.plot(*pca.mean_, 'o', color='black')

c1 = pca.mean_ + pca.components_[0]*pca.explained_variance_[0]
c2 = pca.mean_ + pca.components_[1]*pca.explained_variance_[1]
v1 = np.array([pca.mean_, c1])
v2 = np.array([pca.mean_, c2])

plt.plot(*v1.transpose(), color='black')
plt.plot(*v2.transpose(), color='black')

plt.gca().set_aspect('equal')
plt.show()
