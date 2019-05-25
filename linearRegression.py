import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

FEATURE_SCALING = 1
NORMALIZE = 0b1

class linearRegression:
	def __init__(self, options):
		self.options = FEATURE_SCALING | NORMALIZE
		self.data_test_x = None # matrice
		self.data_test_y = None # vecteur
		self.X = None # matrice
		self.y = None # vecteur
		self.training_set_size = 0;
		self.alpha = 0 # nombre
		self.theta = None # matrice

	def get_values_from_file(self, name, type):
		# recupere et parse les informations du fichier
		# la derniere colone est toujours le y
		# separe le fichier en deux sous ensembles : X et data_test

		training_set_values = [20, 30, 40] # matrice
		training_set_goal = [2000, 3000, 4000]# vecteur

		data_set_values = [10, 50] # matrice
		data_set_goal = [1000, 5000]# vecteur

		self.X = np.matrix([20, 30, 40])
		self.y = np.matrix([2000, 3000, 4000])
		self.training_set_size = len(self.y)

		self.data_test_x = np.matrix([10, 50]) # matrice
		self.data_test_y = np.matrix([1000, 5000]) # vecteur

	def improve_data(self):
		if (self.options & FEATURE_SCALING):
			return
		if (self.options & NORMALIZE):
			return
		return
		# feature scaling
		# normailze value

	def calculate(self):

		return

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
		return

	def __cost_function(self):
		return

	def __gradient_descent(self):
		return

	def __change_theta(self, theta):
		return theta - self.alpha * (1 / self.training_set_size) * sum_func()

	def sum_func(self):
		sum = 0
		for X_elem, y_elem in zip(self.X, self.y):
			sum += (__h0_function(X_elem) - y_elem) * X_elem
		return sum

	def __h0_function(self, X_elem):
		ret = 0
		for theta, x in zip(self.theta, X_elem):
			ret += theta * x
		return ret

	def __linear_regression_function(self, x, theta):
		# tetha 0 + somme des theta n * x n
		# tetha est une liste
		# x est une liste
		return theta0 + theta1 * x





	# def __linear_regression_function(self, x):
	# 	return theta0 + theta1 * x



	def f(self):
		return
