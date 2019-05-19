import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import metrics


# Загружаем MNIST датасет через тнезорфлоу
(X_train,y_train),(X_test,y_test) = tf.keras.datasets.mnist.load_data()

# Изображение квадратное!
dim = X_train[0].shape[0]
n_channels = 1

X_train = X_train[:4000]
y_train = y_train[:4000]

#X_test = X_test[5000:6000]
#y_test = y_test[5000:6000]

# Необходимо перешейпить изображения в 4-D тензор!!!!!
X_train = X_train.reshape(X_train.shape[0],dim,dim,1)
X_test = X_test.reshape(X_test.shape[0],dim,dim,1)

# Загрузка данных с помощью tensorflow 
mnist = input_data.read_data_sets('/tmp/data/')


# Простая реализация сверточной нейронной сети 
# У MNIST датасета все изображения серые, поэтому количество каналов = 1
X = tf.placeholder(tf.float32,shape=(None,dim,dim,1))
y = tf.placeholder(tf.int32,shape=(None))



with tf.name_scope('cnn'):
	cnn1 = tf.layers.conv2d(X,filters=10,kernel_size=[3,3],
							padding='same',activation=tf.nn.relu)

	pooling1 = tf.layers.max_pooling2d(inputs=cnn1,pool_size=[2,2],strides=2)
	cnn2 = tf.layers.conv2d(pooling1,filters=10,kernel_size=[3,3],
								padding='same',activation=tf.nn.relu)
	pooling2 = tf.layers.max_pooling2d(cnn2,2,2)
	# выполняем разглаживание тензора!
	flatten = tf.contrib.layers.flatten(pooling2)
	#flatten = tf.reshape(pooling2,[-1,n_channels*dim*dim])
	# Первый плотный слой сети
	dense1 = tf.layers.dense(inputs=flatten,units=1024,
									activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense1,rate=0.4)
	logits = tf.layers.dense(inputs=dropout,units=10)

with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
													labels=y)
	loss = tf.reduce_mean(xentropy)

with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
	correct = tf.nn.in_top_k(logits,y,1)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

minimal = 0
'''
with tf.Session() as sess:
	init.run()
	for epoch in range(100):
		sess.run(training_op,feed_dict={X:X_train,y:y_train})
		acc_val = accuracy.eval(feed_dict={X:X_test,y:y_test})
		print(acc_val,'This is my accuracy on the test dataset!')
		if acc_val > minimal:
			minimal = acc_val
			save = saver.save(sess,'./final.ckpt')


print(minimal,'This is SAVED CNN')
'''

with tf.Session() as sess:
	saver.restore(sess,'./final.ckpt')
	Z = logits.eval(feed_dict={X:X_test})
	y_pred = np.argmax(Z,axis=1)
	
metrics = metrics.accuracy_score(y_test,y_pred)
print(metrics)






