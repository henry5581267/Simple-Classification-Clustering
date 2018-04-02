# =============================================================================
#                           Hierarchical Clustering
# =============================================================================
from sklearn import cluster
import LoadData as Data
import matplotlib.pyplot as plt
import numpy as np

#rectangle label function
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')


# initial clustering
cluster_result_CPBL = [0, 0, 0, 0]
cluster_result_car = [0, 0, 0, 0]

# =============================================================================
#                               CPBL Data
# =============================================================================
X_CPBL = Data.x_CPBL_data  # Load CPBL data
# Hierarchical Clustering
hclust = cluster.AgglomerativeClustering(
    linkage='complete', affinity='manhattan', n_clusters=4)

# fit input
hclust.fit(X_CPBL)
# Get label for input
cluster_CPBL = hclust.labels_

# Identify Clustering
for i in range(0, len(cluster_CPBL)):
    if cluster_CPBL[i] == 0:
        cluster_result_CPBL[0] += 1
    elif cluster_CPBL[i] == 1:
        cluster_result_CPBL[1] += 1
    elif cluster_CPBL[i] == 2:
        cluster_result_CPBL[2] += 1
    elif cluster_CPBL[i] == 3:
        cluster_result_CPBL[3] += 1

# =============================================================================
#                               Car Data
# =============================================================================
X_car = Data.x_car_data  # Load car data
# Hierarchical Clustering
hclust = cluster.AgglomerativeClustering(
    linkage='complete', affinity='manhattan', n_clusters=4)

# fit input
hclust.fit(X_car)
# Get label for input
cluster_car = hclust.labels_
# Identify Clustering
for i in range(0, len(cluster_car)):
    if cluster_car[i] == 0:
        cluster_result_car[0] += 1
    elif cluster_car[i] == 1:
        cluster_result_car[1] += 1
    elif cluster_car[i] == 2:
        cluster_result_car[2] += 1
    elif cluster_car[i] == 3:
        cluster_result_car[3] += 1
# =============================================================================
#                           Plot Data
# =============================================================================
x = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3']
fig, ax = plt.subplots()
ind = np.arange(len(cluster_result_CPBL))*0.5
width = 0.3/1.5
rects1 = ax.bar(ind+0.1, cluster_result_CPBL, width, color="blue")
ax.set_title("Hierarchical CPBL Players Clustering")
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Cluster')
plt.ylabel('Number')
autolabel(rects1)
plt.show()

x = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3']
fig, ax = plt.subplots()
ind = np.arange(len(cluster_result_car))*0.5
width = 0.3/1.5
rects1 = ax.bar(ind+0.1, cluster_result_car, width, color="blue")
ax.set_title("Hierarchical Car Clustering")
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Cluster')
plt.ylabel('Number')
autolabel(rects1)
plt.show()
