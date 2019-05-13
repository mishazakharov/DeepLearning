import pickle
import numpy as np

class LiteOCR():
	def __init__(self,fn='alpha_weights.pkl',pool_size=2):
		[weights,meta] = pickle.load(open(fn,'rb'),encoding='latin1')
		self.vocab = meta['vocab']
		# Количество строк и столбцов в картинке
		self.img_rows = meta['img_side']; self.img_cols = meta['img_side']
		# Загружаем нашу СНН
		self.CNN = LiteCNN()
		# С нашими весами
		self.CNN.load_weights(weights)
		# Определяем размер pooling матрицы
		self.CNN.pool_size=int(pool_size)

	def predict(self,image):
		print(image.shape)
		# Векторизируем нашу картинку для нашей СНН
		X = np.reshape(image,(1,1,self.img_rows,self.img_cols))
		X = X.astype('float32')
		# Делаем предсказание
		predicted_i = self.CNN.predict(X)
		# Возвращаем предсказанные классы!
		return self.vocab[predicted_i]


class LiteCNN():
	def __init__(self):
		# Место сохранения слоев
		self.layers = []
		# Размер пула для макс-пулинга
		self.pool_size = None

	def load_weights(self,weights):
		assert not self.layers,'Weights can only be loaded once!'
		# Добавляем сохраненную матрицу 
		for k in range(len(weights,keys())):
			self.layers.append(weights['layer_{}'.format(k)])

	def predict(self,X):
		# Высокоуровневая логика сверточной нейроннлй сети данной архитектуры!
		h = self.cnn_layer(X,layer_i=0,border_mode='full'); X=h
		h = self.relu_layer(X); X=h;
		h = self.cnn_layer(X,layer_i=2,border_mode='valid');X=h
		h = self.relu_layer(X);X=h
		h = self.maxpooling_layer(X);X=h
		h = self.dropout_layer(X,.25);X=h
		h = self.flatten_layer(X,layer_i=7);X=h;
		h = self.dense_layer(X,fully,layer_i=10);X=h
		h = self.softmax_layer2D(X); X=h
		max_i = self.classify(X)
		return max_i[0]

	def maxpooling_layer(self,convolved_features):
		nb_features = convolved_features.shape[0]
		nb_images = convolved_features.shape[1]
		conv_dim = convolved_features.shape[2]
		res_dim = int(conv_dim/self.pool_size)

		pooled_features = np.zeros((nb_features,nb_images,res_dim,res_dim))
		for image_i in range(nb_images):
			for feature_i in range(nb_features):
				for pool_row in range(res_dim):
					row_start = pool_row * self.pool_size
					row_end = row_start + self.pool_size
					for pool_col in range(res_dim):
						col_start = pool_col * self.pool_size
						col_end = col_start + self.pool_size
						patch = convolved_features[feature_i,image_i,
										row_start,row_end,col_start,col_end]
						pooled_features[feature_i,
									image_i,pool_row,pool_col] = np.max(patch)
									return pooled_features

	def cnn_layer(self,X,layer_i=0,border_mode='full'):
		features = self.layers[layer_i]['param_0']
		bias = self.layers[layer_i]['param_1']
		# Насколько большой фильтр ? 
		patch_dim = features[0].shape[-1]
		# Как много признаков у нас есть ?
		nb_features = features.shape[0]
		# Насколько большое изображение
		image_dim = X.shape[2] # Пусть оно будет квадратным
		# Р Г Б значения
		image_channels = X.shape[1]
		# Сколько изображений у нас есть ? 
		nb_images = X.shape[0]
		# При border_mode='full' мы получаем на выходе изображение без уменьше
		# -ния размеров. Это означает, что фильтр должен выходить за края 
		# входного изображения на размер фильтра
		# Обычно за краями все заполнено 0-ми.
		if border_mode == 'full':
			conv_dim = image_dim + patch_dim - 1
		# При border_mode='valid' мы получаем выход, который меньше входа,
		# потому что свертка вычисляется только когда input и фильтр полность
		# -ю перекрываются.
		if border_mode == 'valid':
			conv_dim = image_dim - patch_dim + 1

		# Инициализируем наше матрицу признаков(фильтр)
		convolved_features = np.zeros((nb_images,nb_features,
												conv_dim,conv_dim))
		# Затем мы пройдемся по каждой картинке 
		for image_i in range(nb_images):
			# По каждому признаку
			for feature_i in range(nb_features):
				# Инициализируем свернутую картинку пустой!
				convolved_image = np.zeros((conv_dim,conv_dim))
				# Для каждого канала
				for channel in range(image_channels):
					# Извлечем признак из нашей карты признаков
					feature = features[feature_i,channel,:,:]
					# Затем определяем специфическую для нашего канала часть 
					# изображения
					image = X[image_i,channel,:,:]
					# Осуществляем свертку
					convolved_features += self.convolve2d(image,feature,
															border_mode)
					# Добавляем смещение к нашей свернутой картинке!
					convolved_image = convolved_image + bias[feature_i]
					# Добавляем это в список свернутых признаков(learnings)
					convolved_features[image_i,feature_i,:,:] =convolved_image
		return convolved_features

	def dense_layer(self,X,layer_i=0):
		# Инициализируем наши веса и смещения для этого слоя
		W = self.layers[layer_i]['param_0']
		b = self.layers[layer_i]['param_1']
		# Вычислим взвешенную сумму
		output = np.dot(X,W) + b
		return output

	def convolve2d(image,feature,border_mode='full'):
		# Определим размерность тензора 
		image_dim = np.array(image.shape)
		feature_dim = np.array(featrue.shape)

		target_dim = image_dim + feature_dim - 1
		fft_result = (np.fft.fft2(image,target_dim) * 
								np.fft.fft2(feature,target_dim))
		target = np.fft.fft2(fft_result).real
		if border_mode == 'valid':
			valid_dim = image_dim - feature_dim + 1
			if np.any(valid_dim < 1):
				valid_dim = feature_dim - image_dim + 1
				start_i = (target_dim - valid_dim) // 2
				end_i = start_i + valid_dim
				target = target[start_i[0]:end_i[0],start_i[1]:end_i[1]]

		return target

	def relu_layer(x):
		''' РЕЛУ слой. '''
		z = np.zeros_like(x)
		return np.where(x>z,x,z)

	def softmax_layer2D(w):
		''' Эта функция будет считать вероятности каждого лейбла на сэмпле'''
		maxes = np.amax(w,axis=1)
		maxes = maxes.reshape(maxes.shape[0],1)
		e = np.exp(w - maxes)
		dist = e / np.sum(e,axis=1,keepdims=True)
		return dist

	def dropout_layer(X,p):
		''' С лой отклюечния! '''
		retain_prob = 1. - p
		X *= retain_prob
		return X


	def flatten_layer(X):
		''' Трансофрмация тензора для уменьшения размерности. '''
		flatX = np.zeros((X.shape[0],np.prod(X.shape[1:])))
		for i in range(X.shape[0]):
			flatX[i,:] = X[i].flatten(order='C')
		return flatX




