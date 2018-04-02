from sklearn.neighbors import KNeighborsClassifier
import LoadData as Data
import numpy as np
import matplotlib.pyplot as plt  # import matplot library

# =============================================================================
#                           Knn Classifier
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
n = 1
X_CPBL = Data.x_CPBL_data
Y_CPBL = Data.y_CPBL_data
class_CPBL_result = [0, 0, 0, 0]
class_CPBL_result_true = [0, 0, 0, 0]
neigh = KNeighborsClassifier(n_neighbors=n)
neigh.fit(X_CPBL, Y_CPBL)
predict_CPBL = neigh.predict(X_CPBL)

# Identify Class
for i in range(0, len(predict_CPBL[:, 0])):
    index_max = np.argmax(predict_CPBL[i, :])
    index_max_true = np.argmax(Y_CPBL[i, :])
    if index_max == 0:
        class_CPBL_result[0] += 1
    elif index_max == 1:
        class_CPBL_result[1] += 1
    elif index_max == 2:
        class_CPBL_result[2] += 1
    elif index_max == 3:
        class_CPBL_result[3] += 1

    if index_max_true == 0:
        class_CPBL_result_true[0] += 1
    elif index_max_true == 1:
        class_CPBL_result_true[1] += 1
    elif index_max_true == 2:
        class_CPBL_result_true[2] += 1
    elif index_max_true == 3:
        class_CPBL_result_true[3] += 1
# =============================================================================
#                               Car Data
# =============================================================================
X_car = Data.x_car_data
Y_car = Data.y_car_data
class_car_result = [0, 0, 0, 0]
class_car_result_true = [0, 0, 0, 0]
neigh = KNeighborsClassifier(n_neighbors=n)
neigh.fit(X_car, Y_car)
predict_car = neigh.predict(X_car)

# Identify Class
for i in range(0, len(predict_car[:, 0])):
    index_max = np.argmax(predict_car[i, :])
    index_max_true = np.argmax(Y_car[i, :])
    if index_max == 0:
        class_car_result[0] += 1
    elif index_max == 1:
        class_car_result[1] += 1
    elif index_max == 2:
        class_car_result[2] += 1
    elif index_max == 3:
        class_car_result[3] += 1

    if index_max_true == 0:
        class_car_result_true[0] += 1
    elif index_max_true == 1:
        class_car_result_true[1] += 1
    elif index_max_true == 2:
        class_car_result_true[2] += 1
    elif index_max_true == 3:
        class_car_result_true[3] += 1

# =============================================================================
#                           Plot Result
# =============================================================================
x = ['Lamigo', 'Brothers', 'Fubon', 'Union']  # xlabel
fig, ax = plt.subplots()
ind = np.arange(len(class_CPBL_result))*0.5
ind1 = [i+0.2 for i in ind]
width = 0.3/1.5
width2 = 0.3/1.5
rects1 = ax.bar(ind, class_CPBL_result, width, color="blue")
rects2 = ax.bar(ind1, class_CPBL_result_true, width2, color="red")
ax.legend((rects1[0], rects2[0]), ('Train result', 'True result'))
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Class')
plt.ylabel('Number')
ax.set_title("CPBL Players Classifier Knn(n= "+str(n)+")Result")
autolabel(rects1)
autolabel(rects2)


x = ['acc', 'good', 'unacc', 'vgood']  # xlabel
fig, ax = plt.subplots()
ind = np.arange(len(class_car_result))*0.5
ind1 = [i+0.2 for i in ind]
width = 0.3/1.5
width2 = 0.3/1.5
rects1 = ax.bar(ind, class_car_result, width, color="blue")
rects2 = ax.bar(ind1, class_car_result_true, width2, color="red")
ax.legend((rects1[0], rects2[0]), ('Train result', 'True result'))
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Class')
plt.ylabel('Number')
ax.set_title("CAR Classifier Knn(n= "+str(n)+")Result")
autolabel(rects1)
autolabel(rects2)
