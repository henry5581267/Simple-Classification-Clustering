import numpy as np
import Neural_Net_Architecture as dp  # import NN architecture
import LoadData as Data  # import training data
import matplotlib.pyplot as plt  # import matplot library


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')


# Define training data and training label
x_train = Data.x_car_data
y_train = Data.y_car_data
predict = np.zeros((len(x_train[:, 0]), len(x_train[0, :])))

num_points = len(x_train[:, 0])
train_split = 1
train_size = int(train_split*num_points)

class_result = [0, 0, 0, 0]
class_result_true = [0, 0, 0, 0]
batch_size = 100
num_epochs = 3001
train_loss = np.zeros((num_epochs, 1))

for i in range(num_epochs):

    dp.learning_rate = dp.update_learning_rate(i)

    # Training with Batchsize
#    for start, end in zip(range(0, len(x_train), batch_size),
#                          range(batch_size, len(x_train), batch_size)):
#       
    dp.run_train(x_train, y_train)
    update_train_data = i % 100

    train_loss[i] = dp.calc_loss(x_train, y_train)

    if update_train_data == 0:
        print(str(i) + " loss:" + str(train_loss[i]))

# Prediction result
predict = dp.run_infer(x_train)

# Get Classification result
for i in range(0, len(predict[:, 0])):
    index_max = np.argmax(predict[i, :])
    index_max_true = np.argmax(y_train[i, :])
    if index_max == 0:
        class_result[0] += 1
    elif index_max == 1:
        class_result[1] += 1
    elif index_max == 2:
        class_result[2] += 1
    elif index_max == 3:
        class_result[3] += 1

    if index_max_true == 0:
        class_result_true[0] += 1
    elif index_max_true == 1:
        class_result_true[1] += 1
    elif index_max_true == 2:
        class_result_true[2] += 1
    elif index_max_true == 3:
        class_result_true[3] += 1

x = ['acc', 'good', 'unacc', 'vgood']
fig, ax = plt.subplots()
ind = np.arange(len(class_result))*0.5
ind1 = [i+0.2 for i in ind]
width = 0.3/1.5
width2 = 0.3/1.5
rects1 = ax.bar(ind, class_result, width, color="blue")
rects2 = ax.bar(ind1, class_result_true, width2, color="red")
ax.legend((rects1[0], rects2[0]), ('Train result', 'True result'))
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x, minor=False)
plt.xlabel('Class')
plt.ylabel('Number')
ax.set_title("NeurNetwork_Result")
autolabel(rects1)
autolabel(rects2)

fig, ax = plt.subplots()
epoch = range(0, num_epochs)
plt.plot(epoch, train_loss)
plt.xlabel('Epoch')
plt.ylabel('Crossentropy')
ax.set_title("Training Loss(Crossentropy)")
plt.show()

dp.save_model("mynet/Car/Car.ckpt")
