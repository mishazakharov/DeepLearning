import numpy as np
import pandas as pd


class NeuralNetwork():
	''' Класс нейронной сети! '''
	def __init__(self):
		# Не понимаю зачем это
		np.random.seed(1)

		# Мы моделируем один нейрон с 3 входными связями и одним output.
		# Мы присоединяем случайные веса к 3 x 1 матрице, со значениями от -1 до 1
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1

	def _sigmoid(self,x):
		''' 
		Вычисляет сигмоидную функцию от x
		'''
		return 1 / (1 + np.exp(-x))

	def _sigmoid_derivative(self,x):
		'''
		Вычисляет производную сигмоидной функции
		'''
		return x * (1-x)

	def train(self,training_set_inputs,training_set_outputs,
											number_of_iterations):
		'''
		Обучение нейронной сети, посредством алгоритма обратного
		распростронения(подробнее в тетради!)
		'''
		for iteration in range(number_of_iterations):
			# Передаем тренировочный набор в нашу нейронную сеть(1нейрон)
			output = self.think(training_set_inputs)

			# Вычисляем ошибку(между выходом и целевым значением!)
			error = training_set_outputs - output

			# Поправка
			adjustment = np.dot(training_set_inputs.T,
							(error * self._sigmoid_derivative(output)))

			# Поправляем
			self.synaptic_weights += adjustment

	def think(self,inputs):
		'''
		Фактически этот метод производит вычисление выходного сигнала
		нейрона, то есть функция активации от взвешенной суммы
		входных сигналов.
		'''
		return self._sigmoid(np.dot(inputs,self.synaptic_weights))


		
if __name__ == "__main__":

	# Инициализируем однонейронную нейронную сеть
	neural_network = NeuralNetwork()


	# Обучающая выборка. 4 сэмпла по 3 признака в каждом.
	training_set_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_set_outputs = np.array([[0,1,1,0]]).T
	
	'''
	Не работает потому что из array нужно сделать МАТРИЦУ!(без запятых и т.д
	)
	training_set_inputss = pd.read_csv('class.txt')
	training_set_inputs = training_set_inputss.drop('0',axis=1).values[:80]
	training_set_outputs = training_set_inputss['0'].values[:80]

	test_X = training_set_inputss.drop('0',axis=1).values[80:]
	test_y = training_set_inputss['0'].values[80:]
	'''
	# Обучаем нейронную сеть 10000 раз!
	neural_network.train(training_set_inputs,training_set_outputs,10000)
	prediction = neural_network.think(np.array([0,0,0]))
	print(prediction)
	print(training_set_outputs)
	print(training_set_inputs)
	
 