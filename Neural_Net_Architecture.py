import os
import tensorflow as tf
import math
import numpy as np

print("Tensorflow version " + tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.set_random_seed(42)
tf.reset_default_graph()
# ==================== Design the DNN model ==================== #

# neurons of each layers
L = 128
M = 128
N = 128
O = 128

# Input and nominal output
X = tf.placeholder(tf.float32, [None, 6], name="X")
Y_ = tf.placeholder(tf.float32, [None, 4], name="Y")
# variable learning rate
lr = tf.placeholder(tf.float32)

# Hidden layer
W1 = tf.Variable(tf.truncated_normal([6, L], stddev=0.1),name='W1')
B1 = tf.Variable(tf.ones([L])*0.1,name='B1')
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1),name='W2')
B2 = tf.Variable(tf.ones([M])*0.1,name='B2')
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1),name='W3')
B3 = tf.Variable(tf.ones([N])*0.1,name='B3')
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1),name='W4')
B4 = tf.Variable(tf.ones([O])*0.1,name='B4')

# Output layer
WO = tf.Variable(tf.truncated_normal([O, 4], stddev=0.1),name='WO')
BO = tf.Variable(tf.ones([4])*0.1,name='BO')

Y1 = tf.nn.tanh(tf.matmul(X, W1)+B1)
Y2 = tf.nn.tanh(tf.matmul(Y1, W2)+B2)
Y3 = tf.nn.tanh(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.tanh(tf.matmul(Y3, W4) + B4)
Y = tf.nn.softmax(tf.matmul(Y4, WO)+BO)

# Define optimizer
#loss = tf.reduce_sum(tf.abs(Y-Y_))
cross_entropy = -tf.reduce_sum(tf.multiply(Y_,tf.log(Y)))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(lr).minimize(loss)
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# learning rate decay
max_learning_rate = 0.002
min_learning_rate = 0.0001
decay_speed = 100
learning_rate = max_learning_rate


# Define function to be called

def update_learning_rate(epoch):
    return min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch / decay_speed)

def run_train(x_train, y_train):
    sess.run(train_step, feed_dict={X: x_train, Y_: y_train, lr: learning_rate})


def calc_loss(x_train, y_train):
    return sess.run(cross_entropy, feed_dict={X: x_train, Y_: y_train, lr: learning_rate})


def run_infer(input_data):
    return sess.run(Y, feed_dict={X: input_data})

def save_model(path):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path)
    print("Save to path: ", save_path)


