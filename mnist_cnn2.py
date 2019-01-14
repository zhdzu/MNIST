import tensorflow as tf
import numpy as np


'''加载MNIST数据集'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 数据集大小探索
print('Training data shape:',mnist.train.images.shape)  #（55000,784）
print('Test data shape:',mnist.test.images.shape)  # (10000,784)
print('Validation data shape:',mnist.validation.images.shape)  # (5000,784)
print('Training label shape:',mnist.train.labels.shape)  # (55000,10)

# 设置tensorflow对GPU使用按需分配
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

sess = tf.InteractiveSession()



'''构建网络'''
'''初始化权重和偏重'''
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  #正态分布初始化权重值，标准差为0.1
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''卷积层和池化层'''
# 定义卷积层，步长为1，SAME输入输出图片一样大
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 定义池化层，2乘2的池化窗口，步长为2
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 输入和输出的占位符placeholder
x_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 把x_转化成卷积需要的形式
X = tf.reshape(x_,shape=[-1,28,28,1])

# 第一层卷积
# Conv1:32个filter个数，5乘5的卷积核;h_conv1.shape=[-1,28,28,32]
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X,w_conv1) + b_conv1)
# Pool1：最大值池化层2x2 [-1,28,28,28]->[-1,14,14,32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
# Conv2:64个filter个数，5乘5的卷积核;h_conv2.shape=[-1,14,14,64]
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
# Pool2:最大值池化层2x2 [-1,14,14,64]->[-1,7,7,64]
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# 图片尺寸减小到7乘7，加入1024个神经元的全连接层
# 池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# 使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep_prob大小
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
keep_prob = tf.placeholder(tf.float32)    #弃权概率0.0-1.0  1.0表示不使用弃权
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''训练和评估模型'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),axis=1))
# ADAM优化器来做梯度最速下降
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess.run(tf.initialize_all_variables())
# sess.run(tf.global_variables_initializer())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
      # 在feed_dict中加入额外的参数keep_prob来控制dropout比例
    train_accuracy = accuracy.eval(feed_dict={
        x_:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train.run(feed_dict={x_: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x_: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

sess.close()