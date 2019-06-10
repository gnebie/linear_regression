import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import time
import json



# FEATURE_SCALING = 1
# NORMALIZE = 0b10
# SHOW_OPTION = 0b100
#
# NORMALIZE_MIN = 0.0
# NORMALIZE_MAX = 1.0
# NORMALIZE_AVREADGE = (NORMALIZE_MAX - NORMALIZE_MIN) / 2

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


	def update_stockaic_count(self, i, size):
		# print(self.settings["stochastic_gardient"]["stocastique_change"] / size * i)
		if i != 0 and self.settings["stochastic_gardient"]["stocastique_change"] * i % size == 0:
			self.stockaic_count += 1
			# print("update self.stockaic_count  " + str(self.stockaic_count))
			# print(self.stockaic_values[self.stockaic_count])
			if self.stockaic_count >= self.settings["stochastic_gardient"]["stocastique_change"]:
				self.stockaic_count = 0

	def get_value(self):
		return self.values

	def get_inverse_value(self):
		return self.invert_val

	def get_goal(self):
		return self.goal


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
		if (self.stockaic_values_prepared) ==  True:
			return
		self.stockaic_values_prepared = True
		self.stockaic_values = []
		pointeur = 0
		count = 0
		add = self.settings["stochastic_gardient"]["stochastic_gardient_size"]
		# datas need to be shuffled in the init function
		while count < self.settings["stochastic_gardient"]["stocastique_change"]:
			self.stockaic_values.append({"values":self.values[pointeur:add + pointeur],"invert_values":self.values[pointeur:add + pointeur].T, "goal":self.goal[pointeur:add + pointeur]})
			count += 1
			pointeur += add
			if len(self.goal[pointeur:add + pointeur]) == 0:
				pointeur = 0


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
		self.invert_val = self.values.T


class linearRegression:
	def __init__(self, options, settings):

		self.settings = self.set_settings(settings)

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
			if self.settings["extend_values"]["square"]:
				for elem in datas[1:]:
					# print(elem)
					total = elem * elem
					add_datas.append(total)
			# if cube
			if self.settings["extend_values"]["cube"]:
				for elem in datas[1:]:
					add_datas.append(elem * elem * elem)
			# if convolutionals values
			if self.settings["extend_values"]["crossed_values"]:
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

		self.theta = np.array(range(self.X_values_size))
		# random.shuffle(self.theta)
		self.y_prime = range(self.training_set.size)

	def improve_data(self):
		# utile d'en faire une fonction :/
		self.training_set.normalize()
		self.evaluate_set.normalize()
		self.validate_set.normalize()

		self.training_set.prepare_stocastik_value()

	def get_evaluation(self, data_set):

		values = data_set.get_stocastik_value()
		return [ self.__h0_function(x.T).item(0) for x in values ]

	def calculate(self):
		# need to redefine the cost function #pointeur sur fonction
		old_cost = self.__cost_function(self.training_set)
		min_size = self.settings["gardient_descent"]["min_iteration"]
		max_size = self.settings["gardient_descent"]["max_iteration"]
		for i in range(max_size):
			# get the values
			# print(i)

			self.__gradient_descent()
			new_cost = self.__cost_function(self.training_set)
			if old_cost < new_cost : # TODO useful?
				print("non convertion, alpha maybe to big")
				# exit(1)
			elif i > min_size and old_cost >= new_cost and old_cost - new_cost < self.settings["gardient_descent"]["min_delta"]:
				print("what a break!")
				print(old_cost)
				print(new_cost)
				break
			old_cost = new_cost
			if i % 10 == 0:
				self.y_prime = self.get_evaluation(self.training_set)
				self.print_graphe()
				# time.sleep(1)
			self.training_set.update_stockaic_count(i, max_size)
			# update the stockastique
		self.test_set()
		return

	def test_set(self):
		print("cost " + str(self.__cost_function(self.evaluate_set) ))
		print("theta " + str(self.theta ))


	def __cost_function(self, data_set):
		y = data_set.get_stocastik_goal()

		sum = self.__cost_function_square_sum(data_set)
		cost_func = sum / (2 * len(y)) # faire une fonction ? utile ? pas besoin de recalculer chaque fois
		return cost_func

	def __cost_function_square_sum(self, data_set):
		x = data_set.get_stocastik_value()
		y = data_set.get_stocastik_goal()
		z = data_set.get_stocastik_inverse_value()
		sum = 0
		h0 = self.__h0_function(z)
		for hx, y_elem in zip(h0, y):
			sum += (hx - y_elem) * (hx - y_elem)
		return sum

	def __gradient_descent(self):
		tmp_theta = []
		cost_func = self.__cost_function(self.training_set)
		i = 0
		# print("self.theta {0!s} X {1!s}".format(self.theta, self.X.T))
		for theta, x_i in zip(self.theta, self.training_set.get_stocastik_inverse_value()):
			# print("theta {0!s}  x_i {1!s}".format(theta, x_i))
			i += 1
			tmp_theta.append(theta - (self.alpha * self.__gradient_descent_square_sum(x_i) / self.training_set.size))
		# print(tmp_theta)
		# print(self.__cost_function(self.X, self.y))
		# time.sleep(1)

		self.theta = tmp_theta
		return

	def __gradient_descent_square_sum(self, x_i):
		sum = 0
		h0 = self.__h0_function(self.training_set.get_stocastik_inverse_value())
		# print("\n*********************************")
		# for  in x_i.T:
		for x_elem, hx, y_elem in zip(x_i.tolist()[0], h0, self.training_set.get_stocastik_goal()):
			sum += (hx - y_elem) * x_elem
			# print("sum {0!s}  hx {1!s} y {2!s}  x {3!s} ".format(sum, hx, y_elem, x_elem))
		# print("sum {0!s}  h0 {1!s} y_ {2!s}  x_i {3!s} ".format(sum, h0, self.y, x_i.T))

		return sum.item(0)

	def __h0_function(self, x):
		# print("*****************************************************                        ici")
		ret = self.theta * x
		ret = ret.T
		# print("*****************************************************                        la")
		return ret

	def save_model(self):
		# self.min
		# self.max
		# self theta
		# ;
		data = { "min":self.min_val, "max":self.max_val, "theta":self.theta, "settings": self.settings, "X_values_size": self.X_values_size}
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
		self.X_values_size = data["X_values_size"]
		return


	def print_graphe(self):
		for i in range(1, self.X_values_size):
			plt.scatter([a[i] for a in self.training_set.get_stocastik_value().tolist()], self.training_set.get_stocastik_goal())
			plot_list = [a[i] for a in self.training_set.get_stocastik_value().tolist()]
			plot_list2 = plot_list.copy()
			# plot_list.sort(reverse=True)
			plot_second_list = self.y_prime.copy()
			# plot_second_list.sort(reverse=True)
			# plt.plot(plot_list, plot_second_list)
			plt.scatter(plot_list2, self.y_prime)
		plt.draw()
		plt.pause(0.001)
		# plt.show()
		plt.close()

	def print_wait_graphe(self):
		merge = []
		y_prime_validate_set = self.get_evaluation(self.validate_set)
		for i in range(1, self.X_values_size):
			plt.scatter([a[i] for a in self.validate_set.get_value().tolist()], self.validate_set.get_goal())
			# plt.draw()
			plt.plot([a[i] for a in self.validate_set.get_value().tolist()], y_prime_validate_set, 'g')
			# plt.scatter([a[i] for a in self.validate_set.get_value().tolist()], y_prime_validate_set, 'g')
			# plt.pause(0.001)
			plt.show()
		plt.close()

	def print_infos(self):
		self.y_prime = self.get_evaluation(self.training_set)
		# print("     self.data_test_x                 " + str(self.data_test_x))
		# print("     self.data_test_y                 " + str(self.data_test_y))
		# print("     self.X                           " + str(self.X))
		# print("     self.y                           " + str(self.y))
		print("     self.training_set_size           " + str(self.training_set_size))
		print("     self.alpha                       " + str(self.alpha))
		# print("     np.min(self.X)                   " + str(np.min(self.X)))
		# print("     np.max(self.X)                   " + str(np.max(self.X)))
		# print("     np.mean(self.X)                  " + str(np.mean(self.X)))
		# print("     np.min(self.y)                   " + str(np.min(self.y)))
		# print("     np.max(self.y)                   " + str(np.max(self.y)))
		# print("     np.mean(self.y)                  " + str(np.mean(self.y)))
		print("     theta(self.y)                    " + str(self.theta))
		print("     total_min()                      " + str(self.min_val))
		print("     total_max()                      " + str(self.max_val))
		print("     y_prime()                        " + str(self.y_prime))
		# if (self.options & SHOW_OPTION) != 0:
		# 	print(self.options & SHOW_OPTION)
		self.print_wait_graphe()

		return
