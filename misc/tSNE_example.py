from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

np.random.seed(42)
N = 1000
gnr = np.random.randn(N, 3)*[2, 1, 4]
mu1 = [8, 8, 8]
mu2 = [-8, -8, 8]
mu3 = [-8, -8, -8]
mu4 = [8, -8, -8]
cloud1 = mu1 + gnr
cloud2 = mu2 + gnr
cloud3 = mu3 + gnr
cloud4 = mu4 + gnr

cols = ['dodgerblue', 'tomato', 'forestgreen', 'darkorange']
clouds_list = [cloud1, cloud2, cloud3, cloud4]


all_clouds = np.concatenate(clouds_list)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i,c in enumerate(clouds_list):
    ax.scatter(*c.transpose(), alpha=0.3, color=cols[i])
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)
plt.show()

tsne = TSNE(n_components=2)

#tsne.fit(gnr)
#print(tsne.embedding_)

tsne_res = tsne.fit_transform(all_clouds)
#print(tsne_res)

for i in range(len(clouds_list)):
    plt.scatter(*tsne_res[i*N:(i+1)*N].transpose(), alpha=0.3, color=cols[i])

plt.show()
exit()
