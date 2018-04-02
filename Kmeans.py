# =============================================================================
#                           K means Clustering
# =============================================================================

from sklearn.cluster import KMeans  # import kmeans model
import LoadData as Data  # import training data
import numpy as np
import matplotlib.pyplot as plt  # import matplot library

# rectangle label function


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')


# =============================================================================
#                               CPBL Data
# =============================================================================
# initial clustering
cluster_result_CPBL = [0, 0, 0, 0]
cluster_result_car = [0, 0, 0, 0]
# Load CPBL data
X_CPBL = Data.x_CPBL_data
# Kmeans clustering
kmeans = KMeans(n_clusters=4, random_state=40).fit(X_CPBL)
cluster_CPBL = kmeans.labels_

# Identify cluster
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
# Load Car data
X_car = Data.x_car_data
# Kmeans clustering
kmeans = KMeans(n_clusters=4, random_state=40).fit(X_car)
cluster_car = kmeans.labels_
# Identify cluster
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
#                               Plot Data
# =============================================================================

x = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3']
fig, ax = plt.subplots()
ind = np.arange(len(cluster_result_CPBL))*0.5
width = 0.3/1.5
rects1 = ax.bar(ind+0.1, cluster_result_CPBL, width, color="blue")
ax.set_title("Kmeans CPBL Players Clustering")
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
ax.set_title("Kmeans Car Clustering")
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Cluster')
plt.ylabel('Number')
autolabel(rects1)
plt.show()
