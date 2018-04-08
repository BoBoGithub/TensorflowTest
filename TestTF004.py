#--coding=utf8--

import tensorflow as tf

# 第一种 计算
a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
	print "第一种计算:"
	print ("a+b=%d" % sess.run(a+b))
	print ("axb=%d" % sess.run(a*b))
	print  


# 第二种 计算
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
	print "第二种计算:"
	print ("a+b=%d" % sess.run(add, feed_dict={a:2, b:3}))
	print ("axb=%d" % sess.run(mul, feed_dict={a:2, b:3}))
	print 


# 第三种 计算
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
	print "第三种计算:"
	result = sess.run(product)
	print ("matrix1 * matrix2 = %d" % result)
	print
