import tensorflow as tf

# 定义真实结果 x = 1, y = 0
x = tf.constant(1.0, name='input')
y_= tf.constant(0.0, name='corrent_value')

# 设置权重/xy的关系等式/误差计算规则
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')
loss = tf.pow(y - y_, 2, name='loss')

# 设置每次调整幅度 0.0025
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

# 获取Session
sess = tf.Session()

# 初始化图中变量
sess.run(tf.global_variables_initializer())

# 循环100次 调整权重的值
for i in range(100):
	print sess.run(w)
	sess.run(train_step)


# y = wx  => 0 = w*1   -> w越接近0越准确 
