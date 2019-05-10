import tensorflow as tf
import pandas as pd
import numpy as np
from functools import partial

data = pd.read_csv('class.txt')

X = data.drop('0',axis=1)
y = data['0']


# test and train 
X_train,y_train,X_test,y_test = X[:90],y[:90],X[90:],y[90:]

X = tf.placeholder(tf.float32,shape=(None,2),name='X')
training = tf.placeholder_with_default(False,shape=(),name='training')
y = tf.placeholder(tf.int32,shape=(None),name='y')

n_hidden1 = 400
n_hidden2 = 300
n_outputs = 2
n_epochs = 3000

with tf.name_scope('dnn'):
	# Пакетная нормализация на все слои! 
	my_batch_norm_layer = partial(tf.layers.batch_normalization,
									training=training,momentum=0.9)
	hidden1 = tf.layers.dense(X,n_hidden1,name='hidden1')
	bn1 = my_batch_norm_layer(hidden1)
	bn1_act = tf.nn.relu(bn1)
	hidden2 = tf.layers.dense(bn1_act,n_hidden2,name='hidden2')
	bn2 = my_batch_norm_layer(hidden2)
	bn2_act = tf.nn.relu(bn2)
	hidden3 =tf.layers.dense(bn2_act,200,name='hidden3')
	bn3 = my_batch_norm_layer(hidden3)
	bn3_act = tf.nn.relu(bn3)
	hidden4 = tf.layers.dense(bn3_act,100,name='hidden4')
	bn4 = my_batch_norm_layer(hidden4)
	bn4_act = tf.nn.relu(bn4)
	hidden5 = tf.layers.dense(bn4_act,50,name='hidden5')
	bn5 = my_batch_norm_layer(hidden5)
	bn5_act = tf.nn.relu(bn5)
	logits_before_batch = tf.layers.dense(bn5_act,n_outputs,name='outputs')
	logits = my_batch_norm_layer(logits_before_batch)



with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
															logits=logits)
	loss = tf.reduce_mean(xentropy,name='loss')

with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(loss)

with tf.name_scope('evaluate'):
	correct = tf.nn.in_top_k(logits,y,1)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		sess.run(training_op,feed_dict={X:X_train,y:y_train})

	acc_val = accuracy.eval(feed_dict={X:X_test,y:y_test})
	print('Accuracy on the test set is - {}'.format(acc_val))

	accurr = logits.eval(feed_dict={X:X_test})
	y_pred = np.argmax(accurr,axis=1)
	print(y_pred)
	print(y_test,'ACTUALLLLLlllll')





