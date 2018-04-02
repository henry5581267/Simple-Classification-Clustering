from sklearn.linear_model import SGDClassifier
import numpy as np
import LoadData as Data
import matplotlib.pyplot as plt  # import matplot library
# =============================================================================
#                           SGD Classifier
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
#                             CPBL Data
# =============================================================================
X_CPBL = Data.x_CPBL_data
y_CPBL = Data.y_CPBL_data
predict_CPBL = np.zeros((len(y_CPBL[:, 0]), 1))

class_CPBL_result = [0, 0, 0, 0]
class_CPBL_result_true = [0, 0, 0, 0]
Y_CPBL = np.zeros((len(y_CPBL[:, 0]), 1))
for i in range(0, len(Y_CPBL[:, 0])):
    index_max = np.argmax(y_CPBL[i, :])
    if index_max == 0:
        Y_CPBL[i, 0] = 0
    elif index_max == 1:
        Y_CPBL[i, 0] = 1
    elif index_max == 2:
        Y_CPBL[i, 0] = 2
    elif index_max == 3:
        Y_CPBL[i, 0] = 3
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X_CPBL, Y_CPBL)
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.00001,
              eta0=0.0, fit_intercept=True, l1_ratio=0.15,
              learning_rate='optimal', loss='hinge', max_iter=2000, n_iter=300,
              n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
              shuffle=True, tol=0.21, verbose=0, warm_start=False)

predict_CPBL = clf.predict(X_CPBL)

for i in range(0, len(predict_CPBL)):
    if predict_CPBL[i] == 0:
        class_CPBL_result[0] += 1
    elif predict_CPBL[i] == 1:
        class_CPBL_result[1] += 1
    elif predict_CPBL[i] == 2:
        class_CPBL_result[2] += 1
    elif predict_CPBL[i] == 3:
        class_CPBL_result[3] += 1

    if Y_CPBL[i] == 0:
        class_CPBL_result_true[0] += 1
    elif Y_CPBL[i] == 1:
        class_CPBL_result_true[1] += 1
    elif Y_CPBL[i] == 2:
        class_CPBL_result_true[2] += 1
    elif Y_CPBL[i] == 3:
        class_CPBL_result_true[3] += 1

# =============================================================================
#                            Car Data
# =============================================================================
X_car = Data.x_car_data
y_car = Data.y_car_data
Y_car = np.zeros((len(y_car[:, 0]), 1))
predict_car = np.zeros((len(Y_car[:, 0]), 1))
class_car_result = [0, 0, 0, 0]
class_car_result_true = [0, 0, 0, 0]

for i in range(0, len(Y_car[:, 0])):
    index_max = np.argmax(y_car[i, :])
    if index_max == 0:
        Y_car[i, 0] = 0
    elif index_max == 1:
        Y_car[i, 0] = 1
    elif index_max == 2:
        Y_car[i, 0] = 2
    elif index_max == 3:
        Y_car[i, 0] = 3
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X_car, Y_car)
SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.00001,
              eta0=0.0, fit_intercept=True, l1_ratio=0.15,
              learning_rate='optimal', loss='hinge', max_iter=2000, n_iter=300,
              n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
              shuffle=True, tol=0.21, verbose=0, warm_start=False)

predict_car = clf.predict(X_car)

for i in range(0, len(predict_car)):
    if predict_car[i] == 0:
        class_car_result[0] += 1
    elif predict_car[i] == 1:
        class_car_result[1] += 1
    elif predict_car[i] == 2:
        class_car_result[2] += 1
    elif predict_car[i] == 3:
        class_car_result[3] += 1

    if Y_car[i] == 0:
        class_car_result_true[0] += 1
    elif Y_car[i] == 1:
        class_car_result_true[1] += 1
    elif Y_car[i] == 2:
        class_car_result_true[2] += 1
    elif Y_car[i] == 3:
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
ax.set_title("CPBL Players Classifier SGD Result")
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
ax.set_title("CAR Classifier SGD Result")
autolabel(rects1)
autolabel(rects2)
