import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import time
import json



FEATURE_SCALING = 1
NORMALIZE = 0b10
SHOW_OPTION = 0b100

NORMALIZE_MIN = 0.0
NORMALIZE_MAX = 1.0
NORMALIZE_AVREADGE = (NORMALIZE_MAX - NORMALIZE_MIN) / 2

# data goal must be the first colone of a matrice






class dataset:
	def __init__(self, datas, min, max, settings):

		self.settings  = settings
		random.shuffle(datas)

		self.values = np.array([ [1.0] +  elem[1:] for elem in datas], float)
		self.goal = np.array([elem[0] for elem in datas])
		self.invert_val = self.values.T

		self.size = len(self.goal)
		self.values_size = len(self.values[0])


		self.min_val = min
		self.max_val = max

		self.stockaic_count = 0
		self.stockaic_values_prepared = False


	def update_stockaic_count(self):
		self.stockaic_count += 1
		if self.stockaic_count > self.settings["stochastic_gardient"]["stocastique_change"]:
			self.stockaic_count = 0

	def get_stocastik_value(self):
		if self.stockaic_values_prepared == False:
			return self.values
		else:
			return self.stockaic_values[self.stockaic_count]["values"]

	def get_stocastik_inverse_value(self):
		if self.stockaic_values_prepared == False:
			return self.invert_val
		else:
			return self.stockaic_values[self.stockaic_count]["invert_values"]

	def get_stocastik_goal(self):
		if self.stockaic_values_prepared == False:
			return self.goal
		else:
			return self.stockaic_values[self.stockaic_count]["goal"]


	def prepare_stocastik_value(self):
		self.stockaic_values_prepared = True
		self.stockaic_values = []
		pointeur = 0
		count = 0
		add = self.settings["stochastic_gardient"]["stochastic_gardient_size"]
		# datas need to be shuffled in the init function
		while count < self.settings["stochastic_gardient"]["stocastique_change"]:
			self.stockaic_values.append({"values":self.values[pointeur:add + pointeur],"invert_values":self.invert_val[pointeur:add + pointeur], "goal":self.goal[pointeur:add + pointeur]})
			count += 1
			pointeur += add
			if len(self.values[pointeur:add + pointeur]) == 0:
				pointeur = 0
			else:
				pointeur += add

	def normalize(self):
		data = self.values.tolist()
		if (self.settings["normialize"]["status"] == True):
			min_max = [elem1 - elem2 for elem1, elem2 in zip(self.max_val, self.min_val)]
			for x in data:
				for i in range(len(x)):
					if min_max[i] != 0:
						x[i] = ((x[i] - self.min_val[i]) / min_max[i] * self.settings["normialize"]["max"]) - self.settings["normialize"]["min"]
					if x[i] > self.settings["normialize"]["max"]:
						x[i] = self.settings["normialize"]["max"]
					if x[i] < self.settings["normialize"]["min"]:
						x[i] = self.settings["normialize"]["min"]
			if (self.settings["feature_scaling"]):
				data = [x[:1] + [elem - self.settings["normialize"]["avreadge"] for elem in x[1:]] for x in data]
		elif (self.settings["feature_scaling"]):
			for x in data:
				for i in range(len(x)):
					x[i] = x[i] - (self.max_val[i] - self.min_val[i] )
		self.values = np.matrix(data, float)


class linearRegression:
	def __init__(self, options, settings):

		self.settings = self.set_settings(settings)

		self.options =  NORMALIZE | FEATURE_SCALING | SHOW_OPTION
		self.data_test = None
		self.data_train = None
		self.data_test_x = None # matrice
		self.data_test_y = None # vecteur

		self.X = None # matrice
		self.X_values_size = 0
		self.min_val = []
		self.max_val = []
		self.min_val_stop = None
		self.max_val_stop = None
		self.y = None # vecteur
		self.training_set_size = 0;
		self.alpha = settings["alpha"] # nombre
		self.theta = None # matrice
		self.y_prime = None # matrice

	def set_settings(self, setting_file):
		settings = {
			# project name
			"name": "test_0",
			"alpha":0.8,
			"train_evaluate_split":0.8,
			"evaluate_validate_split":0.5,
			"limit_values":{
				"min_status":False,
				"max_status":False,
				"min_limit":[0],
				"max_limit":[0]
			},
			"feature_scaling":True,
			"normialize":{
				"status":True,
				"avreadge":0.0,
				"min":0.0,
				"max":1.0
			},
			"extend_values":{
				"square":True,
				"cube":True,
				"crossed_values":True
			},
			"show_steps":True,
			"stochastic_gardient":{
				"status":True,
				"stochastic_gardient_size": 100,
				"stocastique_change":10
				},
			"regularisation_L2":{ # w au carre / w == teta
				"status":True,
				"lamda": 1.5
				},
			"gardient_descent":{
				"min_iteration":10,
				"max_iteration":200,
				"min_delta":0.0001
			},
			"show_steps":True,
			"show_final":True
		}
		for key in settings.keys():
			if key in setting_file.keys():
				if type(settings[key]) == type(setting_file[key]):
					if type(setting_file[key]) == dict:
						for subkey in settings[key].keys():
							if subkey in setting_file[key].keys():
								if type(settings[key][subkey]) == type(setting_file[key][subkey]) or ((type(settings[key][subkey]) == 'float' or type(settings[key][subkey]) == 'int') and (type(setting_file[key][subkey]) == 'float' or type(setting_file[key][subkey]) == 'int')):
									settings[key][subkey] = setting_file[key][subkey]
					else:
						settings[key] = setting_file[key]
		if settings["train_evaluate_split"] > 1.0 or settings["train_evaluate_split"] < 0.0:
			settings["train_evaluate_split"] = 0.8
		settings["normialize"]["avreadge"] = (settings["normialize"]["max"] - settings["normialize"]["min"]) / 2
		return settings


	def get_file(self, name):
		df = pd.read_table(name,sep = '\t',header = 0)
		print(df.head())
		df.drop_duplicates()
		return df.values.tolist()


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
				# print(elem)
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
		new_data = []
		for elem in datas:
			if len(elem) is not size:
				print("*****************************")
				print("values from datas incorect !")
				print("*****************************")
				raise NameError('Incorect values find in file')
			isnan = False
			for x in elem:
				if math.isnan(x):
					isnan = True
			if isnan is False:
				new_data.append(elem)

		# 	else:
		# 		print("find !")
		# exit(1)
		return new_data

	def get_values_from_file(self, name, type):
		# recupere et parse les informations du fichier
		# la derniere colone est toujours le y
		# separe le fichier en deux sous ensembles : X et data_test
		# raw_datas = self.__init_test() # supprimer les lignes vides
		raw_datas = self.get_file(name) # panda?
		#test the values
		raw_datas = self.__test_the_values(raw_datas)
		self.__prepare_datas_for_training(raw_datas)
		return

	def __get_min_max(self, datas):
		values_size = len(datas[0])
		self.X_values_size = values_size
		min_val = datas[0].copy()
		max_val = datas[0].copy()
		for x in datas:
			for i in range(values_size):
				index = i
				if min_val[index] > x[i]:
					min_val[index] = x[i]
				if self.min_val_stop != None and min_val[index] < self.min_val_stop:
					min_val[index] = self.min_val_stop
				if max_val[index] < x[i]:
					max_val[index] = x[i]
				if self.max_val_stop != None and max_val[index] > self.max_val_stop:
					max_val[index] = self.max_val_stop
		min_val[0] = 1
		max_val[0] = 1
		self.max_val = min_val
		self.min_val = max_val
		return (min_val, max_val)


	def __prepare_datas_for_evaluation(self, raw_datas):
		datas = raw_datas
		datas = self.augment_datas(datas)
		self.evaluate_set = dataset(datas, self.min_val, self.max_val, self.settings)
		self.y_prime = range(self.evaluate_set.size)
		self.evaluate_set.normalize()
		return

	def __prepare_datas_for_training(self, raw_datas):
		datas = raw_datas

		random.shuffle(datas)
		# datas = datas[:1000]
		datas = self.augment_datas(datas)
		# print("__prepare_datas_for_training begin")

		data_size = len(datas)
		min_vals, max_vals = self.__get_min_max(datas)

		training_size = int(data_size * self.settings["train_evaluate_split"])

		evaluate_size = data_size - training_size
		validate_size = int(evaluate_size * self.settings["evaluate_validate_split"])
		evaluate_size -= validate_size

		self.training_set = dataset(datas[:training_size], min_vals, max_vals, self.settings)
		self.evaluate_set = dataset(datas[training_size:training_size + evaluate_size], min_vals, max_vals, self.settings)
		self.validate_set = dataset(datas[training_size + evaluate_size:], min_vals, max_vals, self.settings)

		self.training_set.prepare_stocastik_value()
		# print(self.training_set.get_stocastik_goal())
		# print(self.training_set.get_stocastik_value())
		# print(self.training_set.get_stocastik_inverse_value())

		self.theta = np.array(range(self.X_values_size))
		# random.shuffle(self.theta)
		self.y_prime = range(self.training_set.size)

	def improve_data(self):
		# utile d'en faire une fonction :/
		self.training_set.normalize()
		self.evaluate_set.normalize()
		self.validate_set.normalize()
		# self.X = self.__normalize(self.X)
		# self.data_test_x = self.__normalize(self.data_test_x)
		exit(1)

	# def __normalize(self, datas):
	# 	data = datas.tolist()
	# 	# normailze value
	# 	if (self.options & NORMALIZE):
	# 		min_max = [elem1 - elem2 for elem1, elem2 in zip(self.max_val, self.min_val)]
	# 		# print(min_max)
	# 		# data = [ [ ((x[i] - self.min_val[i]) / min_max[i] * NORMALIZE_MAX) - NORMALIZE_MIN for i in range(len(x)) ] for x in data]
	# 		for x in data:
	# 			for i in range(len(x)):
	# 				if min_max[i] != 0:
	# 					x[i] = ((x[i] - self.min_val[i]) / min_max[i] * NORMALIZE_MAX) - NORMALIZE_MIN
	# 				if x[i] > NORMALIZE_MAX:
	# 					x[i] = NORMALIZE_MAX
	# 				if x[i] < NORMALIZE_MIN:
	# 					x[i] = NORMALIZE_MIN
	# 	# feature scaling
	# 		if (self.options & FEATURE_SCALING):
	# 			data = [x[:1] + [elem - NORMALIZE_AVREADGE for elem in x[1:]] for x in data]
	# 	elif (self.options & FEATURE_SCALING):
	# 		for x in data:
	# 			for i in range(len(x)):
	# 				x[i] = x[i] - (self.max_val[i] - self.min_val[i] )
	# 	return (np.matrix(data, float))

	def get_evaluation(self, values):
		return [ self.__h0_function(x).item(0) for x in values ]

	def calculate(self):
		old_cost = self.__cost_function(self.X, self.y)
		for i in range(200):
			self.__gradient_descent()
			new_cost = self.__cost_function(self.X, self.y)
			old_cost = new_cost
			if i % 10 == 0:
				self.y_prime = self.get_evaluation(self.X)
				self.print_graphe()
			# exit(0)
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
		# self.options (normalise feature_scalings)
		# self.min
		# self.max
		# self theta
		# ;
		data = { "min":self.min_val, "max":self.max_val, "theta":self.theta, "options":self.options, "settings": self.settings, "X_values_size": self.X_values_size}
		try:
			with open(self.settings["name"] + '_save.json', 'w') as outfile:
				json.dump(data, outfile)
		except IOError:
			print("can't write the save file, open error")
		return

	def load_model(self, name):
		try:
			with open(name + '_save.json', 'r') as outfile:
				data = json.load(outfile)
		except IOError:
			print("can't write the save file, open error")
			exit(0)
		self.settings = data["settings"]
		self.alpha = data["settings"]["alpha"]
		self.min_val = data["min"]
		self.max_val = data["max"]
		self.theta = data["theta"]
		self.options = data["options"]
		self.X_values_size = data["X_values_size"]
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
		merge = []
		self.y_prime = self.get_evaluation(self.X)
		for i in range(1, self.X_values_size):
			plt.scatter([a[i] for a in self.X.tolist()], self.y)
			plt.draw()
			plt.plot([a[i] for a in self.X.tolist()], self.y_prime, 'g')
			# plt.pause(0.001)
			plt.show()
		plt.close()

	def print_infos(self):
		self.y_prime = self.get_evaluation(self.X)
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
		print("     total_min()                      " + str(self.min_val))
		print("     total_max()                      " + str(self.max_val))
		print("     y_prime()                        " + str(self.y_prime))
		if (self.options & SHOW_OPTION) != 0:
			print(self.options & SHOW_OPTION)
			self.print_wait_graphe()

		return
