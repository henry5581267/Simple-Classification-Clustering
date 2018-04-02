import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import LoadData as Data
# =============================================================================
#                               Fuzzy C-means
# =============================================================================


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
cluster_result_CPBL = [0, 0, 0, 0]
X_CPBL = (Data.x_CPBL_data).transpose()
fpcs_CPBL = []
for n_cluster_CPBL in range(2, 11):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_CPBL, n_cluster_CPBL, 2,
        error=0.005,
        maxiter=10000000, init=None)
    if n_cluster_CPBL == 4:
        cluster_CPBL = u  # If number of cluster is 4
    # Store fpc values for later
    fpcs_CPBL.append(fpc)

cluster_CPBL = np.transpose(cluster_CPBL)
for i in range(0, len(cluster_CPBL)):
    index_max = np.argmax(cluster_CPBL[i, :])
    if index_max == 0:
        cluster_result_CPBL[0] += 1
    elif index_max == 1:
        cluster_result_CPBL[1] += 1
    elif index_max == 2:
        cluster_result_CPBL[2] += 1
    elif index_max == 3:
        cluster_result_CPBL[3] += 1

# =============================================================================
#                               Car Data
# =============================================================================
cluster_result_car = [0, 0, 0, 0]
X_car = (Data.x_car_data).transpose()  # Load car data
# Hierarchical Clustering
fpcs_car = []
for n_cluster_car in range(2, 11):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_car, n_cluster_car, 2,
        error=0.005,
        maxiter=10000000, init=None)
    if n_cluster_car == 4:  # If number of cluster is 4
        cluster_car = u
    # Store fpc values for later
    fpcs_car.append(fpc)

# Identify Clustering
cluster_car = np.transpose(cluster_car)
for i in range(0, len(cluster_car)):
    index_max = np.argmax(cluster_car[i, :])
    if index_max == 0:
        cluster_result_car[0] += 1
    elif index_max == 1:
        cluster_result_car[1] += 1
    elif index_max == 2:
        cluster_result_car[2] += 1
    elif index_max == 3:
        cluster_result_car[3] += 1

# =============================================================================
#                               Plot Data
# =============================================================================
# CPBL
x = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3']
fig, ax = plt.subplots()
ind = np.arange(len(cluster_result_CPBL))*0.5
width = 0.3/1.5
rects1 = ax.bar(ind+0.1, cluster_result_CPBL, width, color="blue")
ax.set_title("Fuzzy Partition Coefficient CPBL Players")
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Cluster')
plt.ylabel('Number')
plt.savefig('Fuzzy_Cmeans_Result.png')
autolabel(rects1)

# CPBL FPC
num_cluster = range(2, n_cluster_CPBL+1)
fig, ax = plt.subplots()
plt.plot(num_cluster, fpcs_CPBL)
ax.set_title("Fuzzy Partition Coefficient CPBL Players")
plt.xlabel('Number of Cluster')
plt.ylabel('FPC')
plt.savefig('Fuzzy Partition Coefficient.png')
plt.show()

# Car
x = ['cluster 0', 'cluster 1', 'cluster 2', 'cluster 3']
fig, ax = plt.subplots()
ind = np.arange(len(cluster_result_car))*0.5
width = 0.3/1.5
rects1 = ax.bar(ind+0.1, cluster_result_car, width, color="blue")
ax.set_title("Fuzzy Cmeans Clustering Car ")
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Cluster')
plt.ylabel('Number')
plt.savefig('Fuzzy_Cmeans_Result.png')
autolabel(rects1)

# Car FPC
num_cluster = range(2, n_cluster+1)
fig, ax = plt.subplots()
plt.plot(num_cluster, fpcs_car)
ax.set_title("Fuzzy Partition Coefficient Car")
plt.xlabel('Number of Cluster')
plt.ylabel('FPC')
plt.savefig('Fuzzy Partition Coefficient.png')
plt.show()
