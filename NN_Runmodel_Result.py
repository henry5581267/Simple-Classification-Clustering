import tensorflow as tf
import numpy as np
import LoadData as Data
import matplotlib.pyplot as plt  # import matplot library

# Rectangle label function


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
#                           Run Model function
# =============================================================================


def runmodel(x_data, outputsize, modelpath):
    tf.reset_default_graph()  # Reset
    # Number of Neurons of Layers
    L = 128
    M = 128
    N = 128
    O = 128
    W1 = tf.Variable(
        np.zeros((len(x_data[0, :]), L)), dtype=tf.float32, name='W1')
    B1 = tf.Variable(np.zeros(L), dtype=tf.float32, name='B1')
    W2 = tf.Variable(np.zeros((L, M)), dtype=tf.float32, name='W2')
    B2 = tf.Variable(np.zeros(M), dtype=tf.float32, name='B2')
    W3 = tf.Variable(np.zeros((M, N)), dtype=tf.float32, name='W3')
    B3 = tf.Variable(np.zeros(N), dtype=tf.float32, name='B3')
    W4 = tf.Variable(np.zeros((N, O)), dtype=tf.float32, name='W4')
    B4 = tf.Variable(np.zeros(O), dtype=tf.float32, name='B4')

    # Output layer
    WO = tf.Variable(np.zeros((O, outputsize)), dtype=tf.float32, name='WO')
    BO = tf.Variable(np.zeros(outputsize), dtype=tf.float32, name='BO')

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, modelpath)
    # Restore weight
    W1 = sess.run(W1)
    W2 = sess.run(W2)
    W3 = sess.run(W3)
    W4 = sess.run(W4)
    WO = sess.run(WO)

    # Restore biases
    B1 = sess.run(B1)
    B2 = sess.run(B2)
    B3 = sess.run(B3)
    B4 = sess.run(B4)
    BO = sess.run(BO)

    # Run result
    X = np.float32(x_data)
    Y1 = tf.nn.tanh(tf.matmul(X, W1)+B1)
    Y2 = tf.nn.tanh(tf.matmul(Y1, W2)+B2)
    Y3 = tf.nn.tanh(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.tanh(tf.matmul(Y3, W4) + B4)
    Y = tf.matmul(Y4, WO)+BO
    Y = sess.run(Y)

    return Y  # Calculate result


#%%
# =============================================================================
#                                Car Data
# =============================================================================
Y_car = runmodel(Data.x_car_data, 4, "mynet/Car/Car.ckpt")
#%%Identify Class
class_car_result = [0, 0, 0, 0]
class_car_result_true = [0, 0, 0, 0]
for i in range(0, len(Y_car[:, 0])):
    index_max = np.argmax(Y_car[i, :])
    index_max_true = np.argmax(Data.y_car_data[i, :])
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
# %%
# =============================================================================
#                                CPBL Data
# =============================================================================
# %%Training Data
Y_CPBL = runmodel(Data.x_CPBL_data, 4, "mynet/CPBL/CPBL.ckpt")

# %%Identify Class
class_CPBL_result = [0, 0, 0, 0]
class_CPBL_result_true = [0, 0, 0, 0]
for i in range(0, len(Y_CPBL[:, 0])):
    index_max = np.argmax(Y_CPBL[i, :])
    index_max_true = np.argmax(Data.y_CPBL_data[i, :])
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
# %%Custom Data
# =============================================================================
#                           Format PA AVG OBP SLG
# =============================================================================
custom_num_information = np.array([550, 0.414, 0.476, 0.689])
custom_num_information = np.reshape(
    custom_num_information, [1, len(custom_num_information)])
# Normalization
# Numeric information
custom_num_information = (custom_num_information -
                          Data.maxix_CPBL)/(Data.maxix_CPBL-Data.minix_CPBL)
# onehot information
# =============================================================================
#                   Format 13 14 15 16 17 F_goabroad T_goabroad
# =============================================================================
custom_onehot_information = np.array([0, 0, 0, 0, 1, 0, 1])
custom_onehot_information = np.reshape(custom_onehot_information, [
                                       1, len(custom_onehot_information)])
custom_inform = np.concatenate(
    (custom_onehot_information, custom_num_information), axis=1)
custom_inform = np.reshape(custom_inform, [len(custom_inform[0, :]), 1])
custom_inform = np.transpose(custom_inform)
# Convert yo float32
custom_inform = np.float32(custom_inform)
# Calculate Result
custom_predict = runmodel(custom_inform, 4, "mynet/CPBL/CPBL.ckpt")
index_custom = np.argmax(custom_predict)
if index_custom == 0:
    print("Predict in Lamigo")
elif index_custom == 1:
    print("Predict in Brothers")
elif index_custom == 2:
    print("Predict in Fubon")
elif index_custom == 3:
    print("Predict in Union")

# %%
# =============================================================================
#                       Plot Result for Training Data
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
ax.set_title("CPBL Player Classifier NeurNetwork_Result")
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
ax.set_title("CAR Classifier NeurNetwork_Result")
autolabel(rects1)
autolabel(rects2)
