import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv('Admissions.csv')

scaled_housing_data_plus_bias = data.drop('Chance of Admit ',axis=1).values
y = data['Chance of Admit '].values
n=7
m,b = data.shape
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32,name='X')
y = tf.constant(y.reshape(-1,1),dtype=tf.float32,name='y')
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name='theta')

y_pred = tf.matmul(X,theta,name='predictions')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name='mse')
gradients = 2/m * tf.matmul(tf.transpose(X),error)
training_op = tf.assign(theta,theta - learning_rate*gradients)

init = tf.global_variables_initializer()
# УЗЕЛ СОХРАНЕНИЯ МОДЕЛИ
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		if epoch%100 == 0:
			print(epoch,mse.eval())
		sess.run(training_op)
	best_theta = theta.eval()
	# Сохранение методом save!!!!!!
	save_path = saver.save(sess,'model.ckpt') 


'''
TensorFlow предоставляет гибкий API, чтобы автоматически считать градиенты - 
- это называется автоматическим дифференцированием, но также она предоставляет
автоматические оптимизаторы типа градиентнонго спуска!!!
Поэтому вместо gradient и training_op может быть:

gradient = optimizer = tf.train.GradientDescentOptimizer
(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
!

Чтобы сохранить граф нужно создать экземпляр  Saver() и применить метод
save() в том месте, где должно происходить сохранение
Чтобы восставновить граф нужно использовать метод restore() Save() в открытой
сессии. Можно сохранять не граф, а отдельные узлы  через словарь 
{имя:переменная}
Метод tf.train.import_meta_graph(/...model.ckpt.meta) использует созданный meta
файл чтобы добавить сохраненнеый граф к настоящему графу!

'''

