import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# Количество нейронов во входном сквозном слое
n_inputs = 28*28
# Количество нейронов в первом слое
n_hidden1 = 300
# Количество нейронов во втором слое
n_hidden2 = 100
# Количество нейронов в выходном слое
n_outputs = 10

# Создание узлов-заполнителей для хранения X и y(подробнее в книге)
X = tf.placeholder(tf.float32,shape=(None,n_inputs),name='X')
y = tf.placeholder(tf.int64,shape=(None),name='y')

def neuron_layer(X,n_neurons,name,activation=None):
	''' Создает нейронный слой. '''
	n_inputs = int(X.get_shape()[1])
	# Нормальное (гауссово) распределение, помогает градиентному спуску
	# быстрее сходиться. Следующие 2 строки просто обеспечивают создание
	# случайного тензора весов!!! 
	stddev = 2 / np.sqrt(n_inputs + n_neurons)
	init = tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
	# Создание двумерного тензора W.
	W = tf.Variable(init,name='kernel')
	# Вектор смещения
	b = tf.Variable(tf.zeros([n_neurons]),name='bias')
	Z = tf.matmul(X,W) + b
	if activation is not None:
		return activation(Z)
	else:
		return Z

'''
Это ручное создание 
with tf.name_scope('dnn'):
	# Создание 2ух скрытых слоев и одного "логита"!
	hidden1 = neuron_layer(X,n_hidden1,name='hidden1',activation=tf.nn.relu)
	hidden2 = neuron_layer(hidden1,n_hidden2,name='hidden2',
										activation=tf.nn.relu)
	# Логит - это выход нейронной сети до прохождения многопеременной 
	# функции активации(подробонсти в книге)
	logits = neuron_layer(hidden2,n_outputs,name='outputs')
Далее представлено автоматическое создание слоев с помощью 
tf.layers.dense!!!
'''

with tf.name_scope('dnn'):
	hidden1 = tf.layers.dense(X,n_hidden1,name='hidden1',activation=tf.nn.relu)
	hidden2 = tf.layers.dense(hidden1,n_hidden2,name='hidden2',
											activation=tf.nn.relu)
	logits = tf.layers.dense(hidden2,n_outputs,name='outputs')

# Создание подпространства имен loss, в качестве функции издержки используется 
# перекрестная энтропия на логитах!
with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
													logits=logits)
	# Вычисление средней перекрестной энтропии по всем образцам!
	loss = tf.reduce_mean(xentropy,name='loss')

# Осуществление градиентного спуска для обновления весов нашей модели!
learning_rate = 0.01
with tf.name_scope('train'):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

# Оценка
with tf.name_scope('eval'):
	# Вычисляет корректен ли прогноз сети, путем проверки, соответствует ли
	# самый высоки логит целевому классу
	correct = tf.nn.in_top_k(logits,y,1)
	# in_top_k возвращает одномерный тензор, поэтому приводим его к виду с 
	# плавающей точкой и считаем среднее(общая правильность сети!)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Загрузка данных с помощью tensorflow 
mnist = input_data.read_data_sets('/tmp/data/')

n_epochs = 40
# Размер мини-пакета!
batch_size = 50

'''
# Прогонка графа: 
with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples//batch_size):
			X_batch,y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op,feed_dict={X:X_batch,y:y_batch})
		acc_train = accuracy.eval(feed_dict={X:X_batch,y:y_batch})
		acc_val = accuracy.eval(feed_dict={X:mnist.validation.images,
											y:mnist.validation.labels})

		print(epoch, 'Правильность при обучениее:', acc_train,
						'Правильность пр проверке:',acc_val)

	save_path = saver.save(sess,'./my_model_final.ckpt')
Уже прогнал
'''

x,y = mnist.train.next_batch(1)

# Использование модели
with tf.Session() as sess:
	saver.restore(sess,'./my_model_final.ckpt')
	X_new_scaled = x # Ряд новых данных(проскейленых)
	Z = logits.eval(feed_dict={X:X_new_scaled})
	y_pred = np.argmax(Z,axis=1)

print('Predicted - ',y_pred)
print('Actual - ',y)







