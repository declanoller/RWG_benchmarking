from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

np.random.seed(42)
N = 1000
gnr = np.random.randn(N, 3)*[2, 4, 4]
mu1 = [8, 8, 8]
mu2 = [-8, -8, 8]
mu3 = [-8, -8, -8]
mu4 = [8, -8, -8]
cloud1 = mu1 + gnr
cloud2 = mu2 + gnr
cloud3 = mu3 + gnr
cloud4 = mu4 + gnr

cols = ['dodgerblue', 'tomato', 'forestgreen', 'darkorange', 'plum', 'gray', 'pink', 'brown']
clouds_list = [cloud1, cloud2, cloud3, cloud4]
all_clouds = np.concatenate(clouds_list)

# plot 3D points
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i,c in enumerate(clouds_list):
    ax.scatter(*c.transpose(), alpha=0.3, color=cols[i])
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)
plt.show()'''

# This assumes the score starts small and goes up.
clusters = list(range(1,20))
kmeans_list = [KMeans(n_clusters=i, random_state=0) for i in clusters]
kmeans_scores = [k.fit(all_clouds).score(all_clouds) for k in kmeans_list]

ptp = np.ptp(kmeans_scores)
mu = np.mean(kmeans_scores)
print(f'Total score diff of {ptp}')

target_perc = 0.85
target_elbow_score = kmeans_scores[0] + target_perc*ptp

print(f'target elbow score is {target_elbow_score}')

score_distances = (np.array(kmeans_scores) - target_elbow_score)**2/ptp**2
print(f'Score distances: {score_distances}')

best_guess_clusters = np.argmin(score_distances) + 1
print(f'Best guess at clusters is {best_guess_clusters} clusters')
plt.plot(clusters, (np.array(kmeans_scores) - mu)/ptp)
plt.plot(clusters[1:10], score_distances[1:10], 'o', color='black')

plt.xlabel('N clusters')
plt.ylabel('Objective score')
plt.tight_layout()
plt.show()
exit()
kmeans = KMeans(n_clusters=3, random_state=0).fit(all_clouds)


plt.plot(x, p(x), linestyle='dashed', color='tomato')
plt.plot(x, p_1d(x), linestyle='dashed', color='forestgreen')
plt.plot(x, p_2d(x), linestyle='dashed', color='orange')
plt.plot(subset_pts, p_2d(subset_pts), 'o', color='black')




# plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i,c in enumerate(all_clouds):
    ax.scatter(*c, alpha=0.3, color=cols[kmeans.labels_[i]])
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)
plt.show()
exit()
# Do tSNE to map to 2 dimensions
tsne = TSNE(n_components=2)
tsne_res = tsne.fit_transform(all_clouds)
#print(tsne_res)

# plot in 2D
for i in range(len(clouds_list)):
    plt.scatter(*tsne_res[i*N:(i+1)*N].transpose(), alpha=0.3, color=cols[i])

plt.show()
exit()
