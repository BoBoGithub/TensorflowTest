#coding=utf8
import tensorflow as tf
import numpy as np

# 设置真实数据值
x_data = tf.constant(5.0, name='x_data')
y_data = tf.constant(11.0, name='y_data')

# 设置偏移量/权重/xy的关系公式  构造一个线性模型
b = tf.Variable(0.0)
w = tf.Variable(1.0, name='weight')
y = tf.mul(w, x_data, name='output')+b

# 设置 最小化方差
lose = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.025)
train = optimizer.minimize(lose)

# 初始化变量 
init = tf.global_variables_initializer()

# 启动图
sess = tf.Session()
sess.run(init)

# 循环拟合
for step in xrange(0, 201):
	sess.run(train)
	if step % 20 == 0:
		print step, sess.run(w), sess.run(b)

# 结果: w = 2.15385  b = 0.230769  => 5*2.15385+0.230769 = 11.000019
