import pandas as pd
import numpy as np

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def simgoid_derivative(x):
	return x * (1.0 - x)

class NeuralNetwork():
	def __init__(self,x,y):
		self.input = x
		self.y = y
		# Все веса первого слоя!!!
		self.weights1 = np.random.rand(self.input.shape[1],4)
		# Все веса второго слоя!!!
		self.weights2 = np.random.rand(4,1)
		self.output = np.zeros(y.shape)


	def feedforward(self):
		# Проход вперед по первому слою
		self.layer1 = sigmoid(np.dot(self.input,self.weights1))
		# Проход вперед по следующему слою(выходному слою)
		self.output = sigmoid(np.dot(self.layer1,self.weights2))

	def backpropogation(self):
		'''
		'''
		# Обновление веса2
		error = self.y - self.output
		learning_rate = 2 
		d_weights2 = np.dot(self.layer1.T,(learning_rate*error*
									simgoid_derivative(self.output)))

		# Обновление веса1
		# Ошибка на всех нейронах первого слоя
		error1 = np.dot(error,self.weights2.T)
		d_weights1 = np.dot(self.input.T,(learning_rate*error1*
							simgoid_derivative(self.layer1)))

		# Обновление весов!
		self.weights1 += d_weights1
		self.weights2 += d_weights2

	def train(self,number_of_iterations):

		for i in range(number_of_iterations+1):
			nn.feedforward()
			nn.backpropogation()

	def predict(self,new_input):	
		l1 = sigmoid(np.dot(new_input,self.weights1))
		output = sigmoid(np.dot(l1,self.weights2))
		return output



if __name__ == "__main__":

    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])

    y = np.array([[0],[1],[1],[0]])

    nn = NeuralNetwork(X,y)
    nn.train(10000)
    output = nn.output
    print(output)
    prediction = nn.predict(np.array([[0,0,0]]))
    print(prediction)




