# coding=utf8
import tensorflow as tf
import numpy as np

# 设置真实数据
x_data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y_data = tf.constant([3.0, 5.0, 7.0, 9.0, 11.0, 13.0])

# 设置线性模型
b = tf.Variable(0.0)
w = tf.Variable(1.0)
y = tf.mul(w, x_data) + b

# 设置每次调整幅度 0.035
loss	= tf.reduce_mean(tf.square(y - y_data))
train	= tf.train.GradientDescentOptimizer(0.035).minimize(loss)

# 启动图并初始化变量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 循环拟合
for step in xrange(0, 201):
	sess.run(train)
	if step % 20 == 0:
		print step, sess.run(w), sess.run(b)

# w = 2.01281   b = 0.945145  => 1.0*2.01281+0.945145 = 2.957955  | 3.0
