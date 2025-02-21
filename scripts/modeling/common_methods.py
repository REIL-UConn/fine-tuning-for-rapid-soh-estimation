
from pathlib import Path
import numpy as np
from datetime import datetime
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import keras, optuna, pickle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

from scripts.config import dic_available_dataset_info, dir_results
from scripts.data_processing.common_methods import get_available_dataset_ids


def create_MLP_model(n_hlayers:int, n_neurons:int, act_fnc:str, opt_fnc:str, learning_rate:float, input_shape=(101,), output_shape=1) -> keras.models.Sequential:
	"""Builds a Keras neural network model (MLP) using the specified parameters. The model is optimized for accuracy. Make sure model outputs (if multiple target) are normalized, otherwise optimization will be biased towards one target variable.

	Args:
		n_hlayers (int): Number of fully-connected hidden layers
		n_neurons (int): Number of neurons per hidden layer
		act_fnc (str): Activation function to use (\'tanh\', \'relu\', etc)
		opt_fnc (str): {\'sgd\', \'adam\'} Optimizer function to use 
		learning_rate (float): Learning rate
		input_shape (int, optional): Input shape of model. Defaults to (100,).
		output_shape (int, optional): Output shape of model. Default to 7.
	Raises:
		ValueError: _description_

	Returns:
		keras.models.Sequential: compiled Keras model
	"""

	# add input layer to Sequential model
	model = keras.models.Sequential()
	model.add( keras.Input(shape=input_shape) )

	# add hidden layers
	for i in range(n_hlayers):
		model.add( keras.layers.Dense(units=n_neurons, activation=act_fnc) )
		
	# add output layer
	model.add( keras.layers.Dense(output_shape) )

	# compile model with chosen metrics
	opt = None
	if opt_fnc == 'adam':
		opt = keras.optimizers.Adam(learning_rate=learning_rate)
	elif opt_fnc == 'sgd':
		opt = keras.optimizers.SGD(learning_rate=learning_rate)
	else:
		raise ValueError("opt_func must be either \'adam\' or \'sgd\'")

	model.compile(
		optimizer=opt,
		loss=keras.losses.mean_squared_error,      
		# make sure to normalize all outputs, otherwise DCIR values will drastically skew MSE reading compared to error of predicted SOH
		metrics=['accuracy'] )
	return model

class ModelOptimizer:
	"""
		Args:
			features (np.ndarray): An array of features to use as model input (#rows = number of smaples, #cols = number of features)
			targets (np.ndarray): An array of targets to use as model output  (#rows = number of smaples, #cols = number of target)
			splits (np.ndarray): An array of (train_idxs, test_idxs) pairs. Loss will be average over all splits.
			model_to_use (str): The model name to optimize. Currently only supports: {'ridge', 'lasso', 'elasticnet', 'randomforest', 'mlp'}
			n_trials (int): The number of optuna optimization trials
			data_scaler (None): {None, StandardScaler, MinMaxScaler} If not None, features will be scaled.
			loss_fnc (_type_, optional): Can provide a custom loss function. If None, the mean_squared_error function is used.
			random_state (int, optional): Defaults to 1.
		"""
	def __init__(self, features:np.ndarray, targets:np.ndarray, splits:np.ndarray, model_to_use:str, n_trials:int, data_scaler:None, loss_fnc=None, random_state:int=1) -> None:
		"""
		Args:
			features (np.ndArray): An array of features to use as model input (#rows = number of smaples, #cols = number of features)
			targets (np.ndArray): An array of targets to use as model output  (#rows = number of smaples, #cols = number of target)
			splits (np.ndArray): An array of (train_idxs, test_idxs) pairs. Loss will be average over all splits.
			model_to_use (str): The model name to optimize. Currently only supports: {'ridge', 'lasso', 'elasticnet', 'randomforest', 'mlp'}
			n_trials (int): The number of optuna optimization trials
			data_scaler (None): {None, StandardScaler, MinMaxScaler} If not None, features will be scaled.
			loss_fnc (_type_, optional): Can provide a custom loss function. If None, the mean_squared_error function is used.
			random_state (int, optional): Defaults to 1.
		"""

		assert targets.shape[1] == 1, "Currently only a single target variable is supported."
		assert targets.shape[0] == features.shape[0], f"Dimension mismatch. Target has {targets.shape[0]} samples but features has {features.shape[0]} samples."
		self.supported_models = ['ridge', 'lasso', 'elasticnet', 'randomforest', 'mlp']
		self.features = features
		self.targets = targets
		self.splits = splits
		assert model_to_use in self.supported_models, f"{model_to_use} is not currently supported. Allowable models are: {self.supported_models}"
		self.model_to_use = model_to_use
		self.n_trials = n_trials
		self.data_scaler = data_scaler
		self.loss_fnc = mean_squared_error if loss_fnc is None else loss_fnc
		self.random_state = random_state

		self.study = None
		self.study_name = None

	def display_results(self):
		"""Prints the best loss and optimal hyperparameters for the model"""
		if self.study is None:
			message = "ERROR: The model has not been optimized yet. Please run 'optimize_model()'\n"
		else:
			message = '\n' + '*'*100
			message += '\n' + f'  Study: {self.study_name}'
			message += '\n' + '*'*100
			message += '\n' + f'  Best Loss: {self.study.best_trial.value}'
			message += '\n' + '  Best Params: '
			for k,v in self.study.best_trial.params.items():
				message += '\n' +f'    {k}: {v}'
			message += '\n' + '*'*100 + '\n'
		
		print(message)

	def _objective(self, trial:optuna.trial.Trial):
		if self.model_to_use == 'ridge':
			# average the loss over all cross-validation splits
			total_loss = 0
			for train_idxs, test_idxs in self.splits:
				# get suggest alpha value from Optuna search space
				alpha = trial.suggest_float('alpha', 0.0, 1e4)
				# create Ridge model
				model = Ridge(
					alpha=alpha,
					random_state=self.random_state
				)

				X = deepcopy(self.features)
				y = deepcopy(self.targets)
				#region: scale data if specified
				if self.data_scaler is not None:
					scaler_X = self.data_scaler().fit(X[train_idxs])
					scaler_y = self.data_scaler().fit(y[train_idxs])
					X = scaler_X.transform(X)
					y = scaler_y.transform(y)
				#endregion

				# fit model to scaled input and output data
				model.fit(X[train_idxs], y[train_idxs])
				
				# get predictions
				yhat = model.predict(X[test_idxs])

				# add loss to total
				total_loss += self.loss_fnc(y[test_idxs], yhat)

			# return average cross-validation loss
			return total_loss / len(self.splits)

		elif self.model_to_use == 'lasso':
			# average the loss over all cross-validation splits
			total_loss = 0
			for train_idxs, test_idxs in self.splits:
				# get suggested model parameters from Optuna search space
				alpha = trial.suggest_float('alpha', 0, 1.0)

				# create Lasso model
				model = Lasso(
					alpha=alpha,
					random_state=self.random_state
				)

				X = deepcopy(self.features)
				y = deepcopy(self.targets)
				#region: scale data if specified
				if self.data_scaler is not None:
					scaler_X = self.data_scaler().fit(X[train_idxs])
					scaler_y = self.data_scaler().fit(y[train_idxs])
					X = scaler_X.transform(X)
					y = scaler_y.transform(y)
				#endregion

				# fit model to scaled input and output data
				model.fit(X[train_idxs], y[train_idxs])
				
				# get predictions
				yhat = model.predict(X[test_idxs])

				# add loss to total
				total_loss += self.loss_fnc(y[test_idxs], yhat)

			# return average cross-validation loss
			return total_loss / len(self.splits)

		elif self.model_to_use == 'elasticnet':
			# average the loss over all cross-validation splits
			total_loss = 0
			for train_idxs, test_idxs in self.splits:
				# get suggested model parameters from Optuna search space
				alpha = trial.suggest_float('alpha', 0.0, 1e3)
				l1_ratio = trial.suggest_float('l1_ratio', 0.01, 1.0)

				# create ElasticNet model
				model = ElasticNet(
					alpha=alpha, 
					l1_ratio=l1_ratio, 
					random_state=self.random_state,
				)

				X = deepcopy(self.features)
				y = deepcopy(self.targets)
				#region: scale data if specified
				if self.data_scaler is not None:
					scaler_X = self.data_scaler().fit(X[train_idxs])
					scaler_y = self.data_scaler().fit(y[train_idxs])
					X = scaler_X.transform(X)
					y = scaler_y.transform(y)
				#endregion

				# fit model to scaled input and output data
				model.fit(X[train_idxs], y[train_idxs])
				
				# get predictions
				yhat = model.predict(X[test_idxs])

				# add loss to total
				total_loss += self.loss_fnc(y[test_idxs], yhat)

			# return average cross-validation loss
			return total_loss / len(self.splits)

		elif self.model_to_use == 'randomforest':
			# average the loss over all cross-validation splits
			total_loss = 0
			for train_idxs, test_idxs in self.splits:
				# get suggested model parameters from Optuna search space
				n_estimators = trial.suggest_int('n_estimators', 5, 500)
				min_samples_split = trial.suggest_float('min_samples_split', 0.01, 1.0)
				min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.01, 1.0)
				max_features = trial.suggest_float('max_features', 0.01, 1.0)
				max_samples = trial.suggest_float('max_samples', 0.01, 1.0)

				# create RandomForestRegressor model
				model = RandomForestRegressor(
					n_estimators=n_estimators, 
					min_samples_split=min_samples_split,
					min_samples_leaf=min_samples_leaf,
					max_features=max_features,  
					max_samples=max_samples,
					n_jobs=-1,
				)

				X = deepcopy(self.features)
				y = deepcopy(self.targets)
				#region: scale data if specified
				if self.data_scaler is not None:
					scaler_X = self.data_scaler().fit(X[train_idxs])
					scaler_y = self.data_scaler().fit(y[train_idxs])
					X = scaler_X.transform(X)
					y = scaler_y.transform(y)
				#endregion

				# fit model to scaled input and output data
				model.fit(X[train_idxs], y[train_idxs])
				
				# get predictions
				yhat = model.predict(X[test_idxs])

				# add loss to total
				total_loss += self.loss_fnc(y[test_idxs], yhat)

			# return average cross-validation loss
			return total_loss / len(self.splits)

		elif self.model_to_use == 'mlp':
			# average the loss over all cross-validation splits
			total_loss = 0
			iter_count = 0
			for train_idxs, test_idxs in self.splits:
				# get suggested parameter values from Optuna search space
				n_hlayers = trial.suggest_int('n_hlayers', 1, 5)
				n_neurons = trial.suggest_int('n_neurons', 2, 100)
				act_fnc = trial.suggest_categorical('act_fnc', ['relu','sigmoid','softmax','softplus','tanh'])
				opt_fnc = trial.suggest_categorical('opt_fnc', ['adam', 'sgd'])
				learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

				# create sequential NN model
				model = create_MLP_model(
					n_hlayers=n_hlayers,
					n_neurons=n_neurons,
					act_fnc=act_fnc,
					opt_fnc=opt_fnc,
					learning_rate=learning_rate,
					input_shape=self.features.shape[1],
					output_shape=self.targets.shape[1]
				)

				X = deepcopy(self.features)
				y = deepcopy(self.targets)
				#region: scale data if specified
				if self.data_scaler is not None:
					scaler_X = self.data_scaler().fit(X[train_idxs])
					scaler_y = self.data_scaler().fit(y[train_idxs])
					X = scaler_X.transform(X)
					y = scaler_y.transform(y)
				#endregion

				# fit model to scaled input and output data
				early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=False, mode='auto', baseline=None, restore_best_weights=True)
				model.fit(
					X[train_idxs], 
					y[train_idxs],
					callbacks = early_stop, 
					verbose = 0)
				
				# get predictions
				yhat = model.predict(X[test_idxs], verbose=0)
				
				# check if nan value exists in predictions (model failed to converge during traing)
				if not (len(np.where(yhat == np.nan)[0]) == 0) or not (len(np.where(yhat == None)[0]) == 0):
					continue
				# add loss to total
				total_loss += self.loss_fnc(y[test_idxs], yhat)
				iter_count += 1

			# return average cross-validation loss
			return total_loss / iter_count

		else: raise RuntimeError(f"{self.model_to_use} not supported.")

	def optimize_model(self, dir_save:Path=None):
		"""Optimize the model for the specified features, target, and number of trials. Each trial performs an n_split cross-validation and the average loss is used. Optuna study will be saved to dir_save if provided."""
		
		timestamp = datetime.now().strftime("%y%m%d%H%M")
		self.study_name = f"modelOptimization_{self.model_to_use}_{timestamp}"

		if dir_save is not None:
			assert dir_save.exists(), f"Could not find filepath: {dir_save}"
	
		#region: run optimization on specified model
		study = optuna.create_study(
				study_name=self.study_name,
				direction='minimize', 
				sampler=optuna.samplers.TPESampler(seed=self.random_state))
		study.optimize(
			func = self._objective,
			n_trials = self.n_trials,
			n_jobs=-1)
		#endregion

		# save study within this class instance
		self.study = study

		#region: save results to file if specified
		if dir_save is not None:
			pickle.dump(
				study, 
				open(dir_save.joinpath(f"{self.study_name}_study.pkl"), 'wb'),
				protocol=pickle.HIGHEST_PROTOCOL)
			pickle.dump(
				{'features':self.features, 'targets':self.targets, 'splits':self.splits}, 
				open(dir_save.joinpath(f"{self.study_name}_data.pkl"), 'wb'),
				protocol=pickle.HIGHEST_PROTOCOL)
		#endregion

	def get_optimal_model(self):
		"""Returns the model initiallized with the optimal hyperparameters. Note that the retiurned model has not been fit to any data."""
		if self.study is None:
			raise RuntimeError("ERROR: The model has not been optimized yet. Please run 'optimize_model()'\n")
		
		model = None
		if self.model_to_use == 'ridge':
			model = Ridge(**self.study.best_trial.params, random_state=self.random_state)
		elif self.model_to_use == 'lasso':
			model = Lasso(**self.study.best_trial.params, random_state=self.random_state)
		elif self.model_to_use == 'elasticnet':
			model = ElasticNet(**self.study.best_trial.params, random_state=self.random_state)
		elif self.model_to_use == 'randomforest':
			model = RandomForestRegressor(**self.study.best_trial.params, random_state=self.random_state)
		elif self.model_to_use == 'mlp':
			model = create_MLP_model(**self.study.best_trial.params)
		else: raise RuntimeError(f"{self.model_to_use} not supported.")
		return model
		
	def evaluate_model(self, plot:bool=False):
		"""Evaluates the optimal models performance over the specified splits. 

		Args:
			plot (bool, optional): If True, predictions v true are plotted for each split. Defaults to False.
		"""

		total_loss = 0
		fig, ax = (None, None)
		if plot: fig, ax = plt.subplots(figsize=(4,3))
		for i, (train_idxs, test_idxs) in enumerate(self.splits):
			model = self.get_optimal_model()

			X = deepcopy(self.features)
			y = deepcopy(self.targets)
			#region: scale data if specified
			if self.data_scaler is not None:
				scaler_X = self.data_scaler().fit(X[train_idxs])
				scaler_y = self.data_scaler().fit(y[train_idxs])
				X = scaler_X.transform(X)
				y = scaler_y.transform(y)
			#endregion

			# fit model to scaled input and output data
			model.fit(X[train_idxs], y[train_idxs])
			
			# get predictions
			yhat = model.predict(X[test_idxs])

			# add loss to total
			total_loss += self.loss_fnc(y[test_idxs], yhat)

			if plot:
				ax.scatter(yhat, y[test_idxs], label=f"CV {i}")

		if plot:
			ax.set_xlabel("Predicted")
			ax.set_ylabel("True")
			ax.set_title(f"Avg. Loss: {total_loss / len(self.splits)}")
			fig.tight_layout(pad=0.5)
			plt.show()
		else:
			message = '\n' + '*'*100
			message += '\n' + f' {len(self.splits)}-Fold Loss: {total_loss / len(self.splits)}' 
			message += '\n' + '*'*100 + '\n'
			print(message)


def get_default_model_params(model_to_use:str, dataset_id:str):
	assert model_to_use in ['mlp'], "Must be \'mlp\'. Additional models have not been implemented yet"
	param_dic = {}
	if model_to_use == 'mlp':
		param_dic['n_hlayers'] = 5
		param_dic['n_neurons'] = 100
		param_dic['act_fnc'] = 'relu'
		param_dic['opt_fnc'] = 'adam'
		param_dic['learning_rate'] = 1e-3
	return param_dic

def get_optimal_model_params(model_to_use:str, dataset_id:str, pulse_type:str, pulse_soc:int=None):
	"""Load the optimal MLP hyperparameters

	Args:
		model_to_use (str): ['mlp', 'ridge', 'lasso', 'elasticnet', 'randomforest']
		dataset_id (str): _description_
		pulse_type (str): _description_
		pulse_soc (int, optional): _description_. Defaults to None.

	Returns:
		dict: The optimal MLP parameters
	"""
	assert model_to_use in ['mlp', 'ridge', 'lasso', 'elasticnet', 'randomforest']
	assert dataset_id in get_available_dataset_ids()
	assert pulse_type in ['chg', 'dchg']
	assert pulse_soc in dic_available_dataset_info[dataset_id]['pulse_socs_tested'], f"{pulse_soc} not in {dic_available_dataset_info[dataset_id]['pulse_socs_tested']}"
	
	def _get_timestamp(path:Path, model_to_use:str):
		start_idx = str(path.name).rindex(f'modelOptimization_{model_to_use}_') + len(f'modelOptimization_{model_to_use}_')
		end_idx = str(path.name).rindex('_study.pkl')
		return int(str(path.name)[start_idx:end_idx])
	
	soc_key = str(pulse_soc) if pulse_soc is not None else 'all'

	dir_temp = dir_results.joinpath('model_optimization', 'using_relative_voltage', dataset_id, f'{pulse_type}_{soc_key}')
	file_study = sorted(list(dir_temp.glob(f'modelOptimization_{model_to_use}_*_study.pkl')), key=(lambda x: _get_timestamp(x, model_to_use=model_to_use)))[-1]
	assert file_study.is_file()

	file_data = dir_temp.joinpath(f'modelOptimization_{model_to_use}_{_get_timestamp(file_study, model_to_use=model_to_use)}_data.pkl')
	assert file_data.is_file()
	study = pickle.load(open(file_study, 'rb'))
	#endregion
	return study.best_trial.params

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

