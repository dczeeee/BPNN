import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
使用TensorFlow自带算法，然后与自己写的比较
'''
mnist = input_data.read_data_sets('MNIST/', one_hot=True)
# 定义session
sess = tf.InteractiveSession()
# 定义样本空间
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

"""
用随机数列生成的方式，创建含一个隐藏层的神经网络。(784,30,10)
"""
# truncated_normal：选取位于正态分布均值=0.1附近的随机值
w1 = tf.Variable(tf.truncated_normal([784, 30], stddev=0.1))
w2 = tf.Variable(tf.zeros([30, 10]))
b1 = tf.Variable(tf.zeros([30]))
b2 = tf.Variable(tf.zeros([10]))
# 激活函数
L1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
y = tf.nn.sigmoid(tf.matmul(L1, w2) + b2)
# 预测值与真实值的误差
loss = tf.reduce_mean(tf.square(Y - y))
# 梯度下降法:选用GradientDescentOptimizer优化器，学习率为0.3，反向传播最小化loss
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化变量，激活tf框架
tf.global_variables_initializer().run()
# 设置每个批次的大小
batch_size = 500
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
for i in range(10000):
    for batch in range(n_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict=({X: batch_xs, Y: batch_ys}))
        acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print("Iter " + str(i) + ",Testing Accuracy " + str(acc))
