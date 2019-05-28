import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time

FEATURE_SCALING = 1
NORMALIZE = 0b10
SHOW_OPTION = 0b100

NORMALIZE_MIN = 0.0
NORMALIZE_MAX = 1.0
NORMALIZE_AVREADGE = (NORMALIZE_MAX - NORMALIZE_MIN) / 2

# data goal must be the first colone of a matrice

class linearRegression:
	def __init__(self, options):
		self.options =  SHOW_OPTION
		self.options =  NORMALIZE | SHOW_OPTION
		self.options =  NORMALIZE | FEATURE_SCALING | SHOW_OPTION
		self.data_test_x = None # matrice
		self.data_test_y = None # vecteur
		self.data_test_tetra = None # vecteur
		self.data_evaluate_split = 0.8
		self.X = None # matrice
		self.X_values_size = 0
		self.min_X = []
		self.max_X = []
		self.min_X_stop = None
		self.max_X_stop = None
		self.y = None # vecteur
		self.training_set_size = 0;
		self.alpha = 0.50 # nombre
		self.theta = None # matrice
		self.y_prime = None # matrice

	def __init_test(self):
		data_set_goal = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # vecteur
		data_set_values1 = [100, 150, 250, 450, 550, 600, 750, 750, 900, 1000]# matrice
		data_set_values2 = [10, 1500, 250, 450, 590, 100, 750, 750, 900, 1000]# matrice
		data_set_values3 = [100, 150, 250, 450, 550, 6000, 750, 750, 900, 10000]# matrice
		datas = []
		for elem1, elem2, elem3, elem4 in zip(data_set_goal, data_set_values1, data_set_values2, data_set_values3):
			# datas.append([elem1, elem2, elem1 / 10, elem3, elem4])
			datas.append([elem1, elem2])
		return datas

	def augment_datas(self, data_total):
		new_total = []
		for datas in data_total:
			add_datas = []
			# if square
			for elem in datas[1:]:
				print(elem)
				total = elem * elem
				add_datas.append(total)
			# if cube
			for elem in datas[1:]:
				add_datas.append(elem * elem * elem)
			# if convolutionals values
			i = 0
			for elem1 in datas[1:]:
				j = 0
				for elem2 in datas[1:]:
					if j != i:
						add_datas.append(elem1 * elem2)
					j+= 1
				i += 1
			new_total.append(datas + add_datas)
		return new_total

	def __test_the_values(self, datas):
		size = len(datas[0])
		for elem in datas:
			if len(elem) is not size:
				print("*****************************")
				print("values from datas incorect !")
				print("*****************************")
				raise NameError('Incorect values find in file')

		return None

	def get_values_from_file(self, name, type):
		# recupere et parse les informations du fichier
		# la derniere colone est toujours le y
		# separe le fichier en deux sous ensembles : X et data_test
		raw_datas = self.__init_test() # supprimer les lignes vides
		# raw_datas = self.get_file() # panda?
		#test the values
		self.__test_the_values(raw_datas)
		self.__prepare_datas(raw_datas)
		return

	def __get_min_max(self, datas):
		values_size = len(datas[0])
		self.X_values_size = values_size
		min_X = datas[0].copy()
		max_X = datas[0].copy()
		for x in datas:
			for i in range(values_size):
				index = i
				if min_X[index] > x[i]:
					min_X[index] = x[i]
				if self.min_X_stop != None and min_X[index] < self.min_X_stop:
					min_X[index] = self.min_X_stop
				if max_X[index] < x[i]:
					max_X[index] = x[i]
				if self.max_X_stop != None and max_X[index] > self.max_X_stop:
					max_X[index] = self.max_X_stop
		min_X[0] = 1
		max_X[0] = 1
		return (min_X, max_X)


	def __prepare_datas(self, raw_datas):
		datas = []
		for elem in raw_datas:
			if not elem in datas:
				datas.append(elem)
		random.shuffle(datas)
		# datas = self.augment_datas(datas)

		data_size = len(datas)
		self.min_X, self.max_X = self.__get_min_max(datas)

		training_size = int(data_size * self.data_evaluate_split)
		evaluate_size = data_size - training_size

		training_set = datas[:training_size]
		training_set.sort() # TODO usefull?
		training_set_goal = [elem[0] for elem in training_set]
		training_set_values = [elem[1:] for elem in training_set]

		evaluate_set = datas[training_size:]
		evaluate_set.sort() # TODO usefull?
		evaluate_set_goal = [elem[0] for elem in evaluate_set]
		evaluate_set_values = [elem[1:] for elem in evaluate_set]
		# separe the values: training , evaluate

		self.X = np.array([ [1.0] + elem for elem in training_set_values], float)
		self.y = np.array(training_set_goal)
		self.training_set_size = len(self.y)

		self.data_test_x = np.array([ [1.0] + elem for elem in evaluate_set_values], float) # matrice
		self.data_test_y = np.array(evaluate_set_goal) # vecteur

		self.theta = np.array(range(self.X_values_size))
		random.shuffle(self.theta)
		self.y_prime = range(self.training_set_size)

	def improve_data(self):
		self.X = self.__normalize(self.X)
		self.data_test_x = self.__normalize(self.data_test_x)

	def __normalize(self, datas):
		data = datas.tolist()
		# normailze value
		if (self.options & NORMALIZE):
			min_max = [elem1 - elem2 for elem1, elem2 in zip(self.max_X, self.min_X)]
			print(min_max)
			# data = [ [ ((x[i] - self.min_X[i]) / min_max[i] * NORMALIZE_MAX) - NORMALIZE_MIN for i in range(len(x)) ] for x in data]
			for x in data:
				for i in range(len(x)):
					if min_max[i] != 0:
						x[i] = ((x[i] - self.min_X[i]) / min_max[i] * NORMALIZE_MAX) - NORMALIZE_MIN
					if x[i] > NORMALIZE_MAX:
						x[i] = NORMALIZE_MAX
					if x[i] < NORMALIZE_MIN:
						x[i] = NORMALIZE_MIN
		# feature scaling
			if (self.options & FEATURE_SCALING):
				data = [x[:1] + [elem - NORMALIZE_AVREADGE for elem in x[1:]] for x in data]
		elif (self.options & FEATURE_SCALING):
			for x in data:
				for i in range(len(x)):
					x[i] = x[i] - (self.max_X[i] - self.min_X[i] )
		return (np.matrix(data, float))

	def calculate(self):
		old_cost = self.__cost_function(self.X, self.y)
		for i in range(200):
			self.__gradient_descent()
			new_cost = self.__cost_function(self.X, self.y)
			self.y_prime = [ self.__h0_function(x).item(0) for x in self.X ]
			old_cost = new_cost
			if i % 10 == 0:
				self.print_graphe()
		self.test_set()
		return

	def test_set(self):
		print("test : " + str(self.data_test_x))
		print("wait : " + str(self.data_test_y))
		print(self.__h0_function(self.data_test_x))
		print("cost " + str(self.__cost_function(self.data_test_x, self.data_test_y) ))


	def __cost_function(self, x, y):
		sum = self.__cost_function_square_sum(x, y)
		cost_func = sum / (2 * len(y))
		return cost_func

	def __cost_function_square_sum(self, x, y):
		sum = 0
		h0 = self.__h0_function(x)
		for hx, y_elem in zip(h0, y):
			sum += (hx - y_elem) * (hx - y_elem)
		return sum

	def __gradient_descent(self):
		tmp_theta = []
		cost_func = self.__cost_function(self.X, self.y)
		i = 0
		# print("self.theta {0!s} X {1!s}".format(self.theta, self.X.T))
		for theta, x_i in zip(self.theta, self.X.T):
			# print("theta {0!s}  x_i {1!s}".format(theta, x_i))
			i += 1
			tmp_theta.append(theta - (self.alpha * self.__gradient_descent_square_sum(x_i) / self.training_set_size))
		# print(tmp_theta)
		# print(self.__cost_function(self.X, self.y))
		# time.sleep(1)

		self.theta = tmp_theta
		return

	def __gradient_descent_square_sum(self, x_i):
		sum = 0
		h0 = self.__h0_function(self.X)
		# print("\n*********************************")
		# for  in x_i.T:
		for x_elem, hx, y_elem in zip(x_i.tolist()[0], h0, self.y):
			sum += (hx - y_elem) * x_elem
			# print("sum {0!s}  hx {1!s} y {2!s}  x {3!s} ".format(sum, hx, y_elem, x_elem))
		# print("sum {0!s}  h0 {1!s} y_ {2!s}  x_i {3!s} ".format(sum, h0, self.y, x_i.T))

		return sum.item(0)

	def __h0_function(self, x):
		ret = self.theta * x.T
		ret = ret.T
		return ret

	def save_model(self):
		return

	def print_graphe(self):
		for i in range(1, self.X_values_size):
			plt.scatter([a[i] for a in self.X.tolist()], self.y)
			plt.plot([a[i] for a in self.X.tolist()], self.y_prime)
		plt.draw()
		plt.pause(0.001)
		# plt.show()
		plt.close()

	def print_wait_graphe(self):
		for i in range(1, self.X_values_size):
			plt.scatter([a[i] for a in self.X.tolist()], self.y)
			plt.plot([a[i] for a in self.X.tolist()], self.y_prime)
			# plt.draw()
			# plt.pause(0.001)
		plt.show()
		plt.close()

	def print_infos(self):
		print("     self.data_test_x                 " + str(self.data_test_x))
		print("     self.data_test_y                 " + str(self.data_test_y))
		print("     self.X                           " + str(self.X))
		print("     self.y                           " + str(self.y))
		print("     self.training_set_size           " + str(self.training_set_size))
		print("     self.alpha                       " + str(self.alpha))
		print("     np.min(self.X)                   " + str(np.min(self.X)))
		print("     np.max(self.X)                   " + str(np.max(self.X)))
		print("     np.mean(self.X)                  " + str(np.mean(self.X)))
		print("     np.min(self.y)                   " + str(np.min(self.y)))
		print("     np.max(self.y)                   " + str(np.max(self.y)))
		print("     np.mean(self.y)                  " + str(np.mean(self.y)))
		print("     theta(self.y)                    " + str(self.theta))
		print("     total_min()                      " + str(self.min_X))
		print("     total_max()                      " + str(self.max_X))
		print("     y_prime()                        " + str(self.y_prime))
		if (self.options & SHOW_OPTION) != 0:
			print(self.options & SHOW_OPTION)
			self.print_wait_graphe()

		return
