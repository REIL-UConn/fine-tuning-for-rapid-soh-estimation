
import pickle
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import tensorflow as tf
from dataclasses import dataclass, asdict
from typing import Optional, Union, Any
from collections.abc import Generator

from scripts.config import dic_available_dataset_info, dir_results
from scripts.data_processing.common_methods import get_available_dataset_ids, get_processed_data, get_cell_ids_in_group, get_group_ids_from_cells
from scripts.modeling.common_methods import get_default_model_params, get_optimal_model_params, create_MLP_model




fine_tuning_result_keys = {
	'model_names':{
		'source_only':'so',
		'fine_tuning':'ft',
		'target_only':'to',
	},
	'model_error_names':{
		'source_on_source':'so',	# source model on source test data
		'source_on_target':'dt',	# source model on target test data
		'fine_tuning':'ft',			# finetuned model on target test data
		'target_only':'to',			# target model on target test data
	},
	'error_metrics':{
		'mae':mean_absolute_error,
		'rmse':root_mean_squared_error,
		'mape':mean_absolute_percentage_error,
		'r2':r2_score
	}
}

DEFAULT_TL_PARAMS = {
	'source_split_unit':'cell', 
	'source_split_method':'percent', 
	'source_train_size':0.666,
	'source_test_size':0.333, 
	'source_train_soh_bound':0.0, 
	'source_stratify':True,
	'source_normalize':True, 

	'target_split_unit':'cell', 
	'target_split_method':'count', 
	'target_train_size':4,
	'target_test_size':60,
	'target_train_soh_bound':0.0, 
	'target_stratify':False,
	'target_normalize':True,

	'ft_n_retrain_layers':1, 
	'ft_learning_rate_1':1e-3, 
	'ft_learning_rate_2':1e-4,
	'ft_epochs_1':200, 
	'ft_epochs_2':20, 
	'ft_batch_size_1':100, 
	'ft_batch_size_2':1,
	'ft_val_split_1':0.1, 
	'ft_val_split_2':0.1,
}

OPTIMAL_TL_PARAMS = {
	'UConn-ILCC-LFP':{
		'chg':{
			'source_split_unit':'cell', 
			'source_split_method':'percent', 
			'source_train_size':0.666,
			'source_test_size':0.333, 
			'source_train_soh_bound':0.0, 
			'source_stratify':True,
			'source_normalize':True, 

			'target_split_unit':'cell', 
			'target_split_method':'count', 
			'target_train_size':5,			# from TargetTrainSize results (optimal = 5, default was 4)
			'target_test_size':59,
			'target_train_soh_bound':0.0, 
			'target_stratify':False,
			'target_normalize':True,

			'ft_n_retrain_layers':1, 		# from FineTuningLayers results (no change over tested range, leave as default)
			'ft_learning_rate_1':1e-3, 		# from FineTuningLearningRate1 results (no change over tested range, leave as default)
			'ft_learning_rate_2':1e-3,		# from FineTuningLearningRate2 results (optimal = 1e-3, default was 1-e4)
			'ft_epochs_1':200, 
			'ft_epochs_2':20, 				# from FineTuningEpochs2 results (no change over tested range, leave as default)
			'ft_batch_size_1':100, 
			'ft_batch_size_2':1,
			'ft_val_split_1':0.1, 
			'ft_val_split_2':0.1,
		},
		'dchg':{
			'source_split_unit':'cell', 
			'source_split_method':'percent', 
			'source_train_size':0.666,
			'source_test_size':0.333, 
			'source_train_soh_bound':0.0, 
			'source_stratify':True,
			'source_normalize':True, 

			'target_split_unit':'cell', 
			'target_split_method':'count', 
			'target_train_size':5,			# from TargetTrainSize results (optimal = 5, default was 4)
			'target_test_size':59,
			'target_train_soh_bound':0.0, 
			'target_stratify':False,
			'target_normalize':True,

			'ft_n_retrain_layers':1, 		# from FineTuningLayers results (no change over tested range, leave as default)
			'ft_learning_rate_1':1e-3, 		# from FineTuningLearningRate1 results (no change over tested range, leave as default)
			'ft_learning_rate_2':1e-3,		# from FineTuningLearningRate2 results (optimal = 1e-3, default was 1-e4)
			'ft_epochs_1':200, 
			'ft_epochs_2':20, 				# from FineTuningEpochs2 results (no change over tested range, leave as default)
			'ft_batch_size_1':100, 
			'ft_batch_size_2':1,
			'ft_val_split_1':0.1, 
			'ft_val_split_2':0.1,
		},
	},
	'UConn-ILCC-NMC':{
		'chg':{
			'source_split_unit':'cell', 
			'source_split_method':'percent', 
			'source_train_size':0.666,
			'source_test_size':0.333, 
			'source_train_soh_bound':0.0, 
			'source_stratify':True,
			'source_normalize':True, 

			'target_split_unit':'cell', 
			'target_split_method':'count', 
			'target_train_size':3,			# from TargetTrainSize results (optimal = 3, default was 4)
			'target_test_size':41,
			'target_train_soh_bound':0.0, 
			'target_stratify':False,
			'target_normalize':True,

			'ft_n_retrain_layers':1, 		# from FineTuningLayers results (no change over tested range, leave as default)
			'ft_learning_rate_1':1e-3, 		# from FineTuningLearningRate1 results (no change over tested range, leave as default)
			'ft_learning_rate_2':1e-4,		# from FineTuningLearningRate2 results (optimal = 1e-4, default was 1-e4)
			'ft_epochs_1':200, 
			'ft_epochs_2':30, 				# from FineTuningEpochs2 results (optimal = 30, default was 20)
			'ft_batch_size_1':100, 
			'ft_batch_size_2':1,
			'ft_val_split_1':0.1, 
			'ft_val_split_2':0.1,
		},
		'dchg':{
			'source_split_unit':'cell', 
			'source_split_method':'percent', 
			'source_train_size':0.666,
			'source_test_size':0.333, 
			'source_train_soh_bound':0.0, 
			'source_stratify':True,
			'source_normalize':True, 

			'target_split_unit':'cell', 
			'target_split_method':'count', 
			'target_train_size':3,			# from TargetTrainSize results (optimal = 3, default was 4)
			'target_test_size':41,
			'target_train_soh_bound':0.0, 
			'target_stratify':False,
			'target_normalize':True,

			'ft_n_retrain_layers':1, 		# from FineTuningLayers results (no change over tested range, leave as default)
			'ft_learning_rate_1':1e-3, 		# from FineTuningLearningRate1 results (no change over tested range, leave as default)
			'ft_learning_rate_2':1e-4,		# from FineTuningLearningRate2 results (optimal = 1e-4, default was 1-e4)
			'ft_epochs_1':200, 
			'ft_epochs_2':30, 				# from FineTuningEpochs2 results (optimal = 30, default was 20)
			'ft_batch_size_1':100, 
			'ft_batch_size_2':1,
			'ft_val_split_1':0.1, 
			'ft_val_split_2':0.1,
		},
	},
}


@dataclass
class Splitting_Parameters:
	'''
	split_unit : {'cell', 'group'}
	\tWhether the data should be split by cell_id or group_id
	split_method : {'items', 'percent'}
	\tHow the train_size should be applied; a number of items (eg cells) or a percent of all items
	train_size : float 
	\tAmount of training data (uses units and method of 'split_unit' and 'split_method')
	test_size : float 
	\tAmount of test data (uses units and method of 'split_unit' and 'split_method'). If left at 0 will use total amount less train_size
	train_soh_bound : float between 0.0 (inclusive) and 1.0 (exclusive)
	\tThe SOH at which training data should be limited. Ex, for =0.8, the training data will only conists of data >80% SOH
	stratify : bool, default False
	\tCan optionally ensure that train and test sets are each representative of full dataset. Note that stratify can only be True if split_unit=cell
	normalize : bool, default True
	\tWhether to normalize the pulse voltage (creates additional keys in the returned dictionary to store the normalized voltage and the scaler)
	train_ids: list[int], default None
	\tCan optionally specify the exact ids (cell or group) to use for the train set. If empty, a random split is performed using the above parameters.
	test_ids: list[int], default None
	\tCan optionally specify the exact ids (cell or group) to use for the test set. If empty, a random split is performed using the above parameters.
	'''

	split_unit:str
	split_method:str = 'count'
	train_size:float = 0
	test_size:float = 0
	train_soh_bound:float = 0.0
	stratify:bool = False
	normalize:bool = True
	train_ids: list[int] = None
	test_ids: list[int] = None


	def __post_init__(self):
		assert self.split_unit in ['cell', 'group'], "split_unit must be either \'cell\' or \'group\'"
		assert self.split_method in ['count', 'percent'], "split_method must be either \'count\' or \'percent\'"
		if self.train_ids is None:
			if self.split_method == 'count':
				self.train_size = round(self.train_size)
				assert self.train_size > 0, "train_size must be greater than zero"
			else:
				assert isinstance(self.train_size, float) and self.train_size > 0.0 and self.train_size < 1.0, "train_size must be between 0 and 1"
		assert self.train_soh_bound >= 0.0 and self.train_soh_bound < 1.0, "train_soh_bound must be between 0 and 1"
		if self.stratify: assert self.split_unit == 'cell', "Cannot stratify with \'split_unit\' equal to \'group\'. Please set \'split_unit\' to \'cell\' or set \'stratify\' to False"

		if self.train_ids is not None: assert self.test_ids is not None, "If using \'train_ids\' you must also specify \'test_ids\'"
		if self.test_ids is not None: assert self.train_ids is not None, "If using \'test_ids\' you must also specify \'train_ids\'"
			
	def dict(self):
		dic = {k: v for k,v in asdict(self).items()}
		return 

@dataclass
class FineTuning_Parameters:
	'''Parameters for fine-tuning a source model\n'''

	n_retrain_layers: int  = 1
	learning_rate_1: float  = 1e-3
	learning_rate_2: float  = 1e-4
	epochs_1: int  = 200
	epochs_2: int  = 10
	batch_size_1: int   = 100
	batch_size_2: int   = 1
	val_split_1: float = 0.1
	val_split_2: float = 0.1
	
	percent_of_source_train_samples_to_use: float = 0.0		
	target_to_source_sample_weight: float = 1.0

	def assert_in_range(self, val, min=0.0, max=1.0):
		assert val >= min and val <= max

	def __post_init__(self):
		# last function called during initiallization
		self.assert_in_range(self.percent_of_source_train_samples_to_use, 0.0, 1.0)
		self.assert_in_range(self.val_split_1, 0.0, 1.0)
		self.assert_in_range(self.val_split_2, 0.0, 1.0)

	def dict(self):
		return {k: v for k,v in asdict(self).items()}

@dataclass
class TransferLearning_Parameters:
	'''Parameters for transferlearning between one SOC to another within the same chemistry\n'''
	dataset_id: str
	pulse_type: str
	source_soc: int
	target_soc: int
	
	source_target_split_params: Optional[Splitting_Parameters]
	source_train_split_params: Splitting_Parameters
	target_train_split_params: Splitting_Parameters
	ft_params: FineTuning_Parameters

	n_neurons: Optional[int] = None
	n_hlayers: Optional[int] = None
	rand_seed: int = -1
	feature_id:str = 'full_voltage'

	def __post_init__(self):
		assert self.dataset_id in get_available_dataset_ids()
		assert self.pulse_type in ['chg', 'dchg', 'combined']
		if hasattr(self.source_soc, '__len__'):
			for s in self.source_soc: 
				assert s in dic_available_dataset_info[self.dataset_id]['pulse_socs_tested']
		else:
			assert self.source_soc in dic_available_dataset_info[self.dataset_id]['pulse_socs_tested']

		if hasattr(self.target_soc, '__len__'):
			for s in self.target_soc: 
				assert s in dic_available_dataset_info[self.dataset_id]['pulse_socs_tested']
		else:
			assert self.target_soc in dic_available_dataset_info[self.dataset_id]['pulse_socs_tested']

	def __repr__(self):
		message = f"{self.dataset_id}_{self.pulse_type}_{self.source_soc}->{self.target_soc}"
		return message

	def dict(self):
		return {k: v for k,v in asdict(self).items()}


	def as_dataframe(self):
		dic = self.dict()
		source_target_dic = dic.pop('source_target_split_params')
		source_dic = dic.pop('source_train_split_params')
		target_dic = dic.pop('target_train_split_params')
		ft_dic = dic.pop('ft_params')
		formatted_dic = None
		if source_target_dic is not None:
			formatted_dic = {**{k:[v] for k,v in dic.items()}, \
							**{f'source_target_{k}': [v] for k,v in source_target_dic.items()}, \
							**{f'source_{k}': [v] for k,v in source_dic.items()}, \
							**{f'target_{k}': [v] for k,v in target_dic.items()}, \
							**{f'ft_{k}': [v] for k,v in ft_dic.items() }  }
		else:
			formatted_dic = {**{k:[v] for k,v in dic.items()}, \
						**{f'source_{k}': [v] for k,v in source_dic.items()}, \
						**{f'target_{k}': [v] for k,v in target_dic.items()}, \
						**{f'ft_{k}': [v] for k,v in ft_dic.items() }  }
		df = pd.DataFrame(formatted_dic)
		return convert_TLasDF_datatypes(df)

def convert_TLasDF_datatypes(df:pd.DataFrame):
	#region: ensure dataframe columns have proper datatype
	int_cols = [
		'rand_seed', 'ft_n_retrain_layers', 'ft_epochs_1', 'ft_epochs_2', 'ft_batch_size_1', 'ft_batch_size_2']
	float_cols = np.hstack([
		'source_target_train_size', 'source_target_test_size', 'source_target_train_soh_bound', 
		'source_train_size', 'source_test_size', 'source_train_soh_bound', 
		'target_train_size', 'target_test_size', 'target_train_soh_bound',
		'ft_learning_rate_1', 'ft_learning_rate_2', 'ft_val_split_1', 'ft_val_split_2',])
	str_cols = np.hstack([
		'dataset_id', 'pulse_type', 'source_soc', 'target_soc',
		'source_target_split_unit', 'source_target_split_method', 'source_target_train_ids', 'source_target_test_ids',
		'source_split_unit', 'source_split_method', 'source_train_ids', 'source_test_ids',
		'target_split_unit', 'target_split_method', 'target_train_ids', 'target_test_ids',
		'feature_id'])
	bool_cols = np.hstack([
		'source_target_stratify', 'source_target_normalize',
		'source_stratify', 'source_normalize', 
		'target_stratify', 'target_normalize',
	])
	df[int_cols] = df[int_cols].astype(int)
	df[float_cols] = df[float_cols].astype(float)
	df[str_cols] = df[str_cols].astype(str)
	df[bool_cols] = df[bool_cols].astype(bool)

	return df

def str_list_to_list(string, dtype=int):
	temp = string
	if ',' in temp: temp = temp.replace(',', '')
	temp = temp.strip('][').split(' ')
	temp = [x for x in temp if not x == '']
	return np.asarray(temp).astype(int)


def dataset_train_test_split(dataset, params:Splitting_Parameters, suppress_warning=True, rand_seed:int=-1):
	'''
	Splits the provided dataset in a train and test set based on the given Splitting_Parameters

	dataset : dictionary 
	\tMust have following keys: {'features', 'targets', 'cell_id', 'group_id', 'soh', 'soc'}
	params : Splitting_Parameters
	\tDefines how the train and test datasets should be split
	suppress_warning : bool, default True
	\tSuppresses stratification warning (can't stratify if training size is too small)

	Returns: a tuple of dictionaries (train_dict, test_dict)
	'''
	for k in ['features', 'targets', 'cell_id', 'group_id', 'soh', 'soc']:
		assert k in list(dataset.keys())

	# features from source and target datasets to use as the model input and output
	model_input_key = 'features'
	model_output_key = 'targets'

	# set random seed / create random generator
	if rand_seed == -1: rand_seed = None
	rng = np.random.default_rng(seed=rand_seed)

	# masks for train and test sets
	train_mask = None
	test_mask = None

	# split by cell_id
	if params.split_unit == 'cell':
		train_cells = None
		all_cell_ids = np.unique(dataset['cell_id'])
		n_pick_train = None
		n_pick_test = None

		if params.train_ids is not None:
			# use only the specified cell ids for train and test sets
			temp = [dataset['cell_id'] == c for c in params.train_ids]
			train_mask = temp[0]
			for i in range(1, len(temp)):
				train_mask = train_mask | temp[i]

			temp = [dataset['cell_id'] == c for c in params.test_ids]
			test_mask = temp[0]
			for i in range(1, len(temp)):
				test_mask = test_mask | temp[i]

		else:
			# split by number of cells (eg. there should be 'params.train_size' cells in the training set)
			if params.split_method == 'count':
				assert params.train_size < len(all_cell_ids), f"train_size must be less than the total number of cells ({len(all_cell_ids)})"
				n_pick_train = params.train_size
				if params.test_size == 0:
					n_pick_test = len(all_cell_ids) - n_pick_train
				else:
					assert params.test_size <= len(all_cell_ids) - n_pick_train, f"train_size ({params.train_size}) + test_size ({params.test_size}) must be equal to or less than the total number of cells ({len(all_cell_ids)})"
					n_pick_test = params.test_size

			# split by percent of cells (eg. there should be 'params.train_size' % of all cells in the training set)
			else:
				n_pick_train = round(params.train_size * len(all_cell_ids))
				assert n_pick_train > 0, f"The train_size is too small. The smallest allowable train size for this split_unit is {1/len(all_cell_ids)*100}%"
			
				if params.test_size == 0:
					n_pick_test = len(all_cell_ids) - n_pick_train
				else:
					n_pick_test = round(params.test_size * len(all_cell_ids))
					assert n_pick_test + n_pick_train == len(all_cell_ids)
					assert params.test_size <= len(all_cell_ids) - n_pick_train, f"train_size + test_size must be equal to or less than 1.0)"
				
			# stratify cell selection (if specified)
			if params.stratify:
				n_groups = len(np.unique(dataset['group_id']))
				if n_pick_train < n_groups and not suppress_warning:
					print("WARNING: The current train size is smaller than the number of groups. Cannot properly stratify.")
				# create pandas dataframe from cell ids and groups ids to use builtin groupby method
				df = pd.DataFrame({'cell_id':dataset['cell_id'], 'group_id':dataset['group_id']})
				train_cells = []
				while len(train_cells) < n_pick_train:
					# get df of cell & group ids that are not already in train_cells
					df_temp = df.loc[np.logical_not(df['cell_id'].isin(train_cells))]
					# select a new cell from each group
					new_cells = df_temp.groupby('group_id', group_keys=False).sample(n=1, replace=False, random_state=rand_seed)['cell_id'].values
					rng.shuffle(new_cells)
					# add cells to group
					for nc in new_cells: 
						if len(train_cells) < n_pick_train: train_cells.append(nc)
			else:
				train_cells = rng.choice(all_cell_ids, size=n_pick_train, replace=False)

			remaining_cells = np.asarray([c for c in all_cell_ids if c not in train_cells])
			test_cells = rng.choice(remaining_cells, size=n_pick_test, replace=False)

			# create train and test masks
			temp = [dataset['cell_id'] == c for c in train_cells]
			train_mask = temp[0]
			for i in range(1, len(temp)):
				train_mask = train_mask | temp[i]
			if params.test_size == 0:
				test_mask = np.logical_not(train_mask) 
			else:
				temp = [dataset['cell_id'] == c for c in test_cells]
				test_mask = temp[0]
				for i in range(1, len(temp)):
					test_mask = test_mask | temp[i]

			# Store which cell ids were randomly selected 
			params.train_ids = train_cells
			params.test_ids = list(np.unique(dataset['cell_id'][test_mask]))

	# split by group_id
	else:
		train_groups = None
		all_group_ids = np.unique(dataset['group_id'])
		n_pick_train = None
		n_pick_test = None

		if params.train_ids is not None:
			# use only the specified cell ids for train and test sets
			temp = [dataset['group_id'] == c for c in params.train_ids]
			train_mask = temp[0]
			for i in range(1, len(temp)):
				train_mask = train_mask | temp[i]

			temp = [dataset['group_id'] == c for c in params.test_ids]
			test_mask = temp[0]
			for i in range(1, len(temp)):
				test_mask = test_mask | temp[i]

		else:
			# split by number of groups (eg. there should be 'params.train_size' groups in the training set)
			if params.split_method == 'count':
				assert params.train_size < len(all_group_ids), f"train_size must be less than the total number of groups ({len(all_group_ids)})"
				n_pick_train = params.train_size
			   
				if params.test_size == 0:
					n_pick_test = len(all_group_ids) - n_pick_train
				else:
					assert params.test_size <= len(all_group_ids) - n_pick_train, f"train_size + test_size must be equal to or less than the total number of groups ({len(all_group_ids)})"
					n_pick_test = params.test_size

			# split by percent of groups (eg. there should be 'params.train_size' % of all groups in the training set)
			else:
				n_pick_train = int(params.train_size * len(all_group_ids))
				assert n_pick_train > 0, f"The train_size is too small. The smallest allowable train size for this split_unit is {1/len(all_group_ids)*100}%"

				if params.test_size == 0:
					n_pick_test = len(all_group_ids) - n_pick_train
				else:
					n_pick_test = int(params.test_size * len(all_group_ids))
					assert params.test_size <= len(all_group_ids) - n_pick_train, f"train_size + test_size must be equal to or less than 1.0"
					

			train_groups = rng.choice(all_group_ids, size=n_pick_train, replace=False)

			remaining_groups = np.asarray([g for g in all_group_ids if g not in train_groups])
			test_groups = rng.choice(remaining_groups, size=n_pick_test, replace=False)

			temp = [dataset['group_id'] == g for g in train_groups]
			train_mask = temp[0]
			for i in range(1, len(temp)):
				train_mask = train_mask | temp[i]
			if params.test_size == 0:
				test_mask = np.logical_not(train_mask)
			else:
				temp = [dataset['group_id'] == g for g in test_groups]
				test_mask = temp[0]
				for i in range(1, len(temp)):
					test_mask = test_mask | temp[i]


			# Store which groups ids were randomly selected 
			params.train_ids = train_groups
			params.test_ids = list(np.unique(dataset['group_id'][test_mask]))

	#region: limit training data to early life if specified
	# soh may be reported as % or decimal. Force into decimal form (ie, between 0 and 1)
	if not ((np.max(dataset['soh']) <= 1.0) and (np.min(dataset['soh']) >= 0.0)):
		dataset['soh'] /= 100.0
	if params.train_soh_bound > np.min(dataset['soh']):
		# find all values where q_dchg < train_soh_bound
		above_soh_mask = dataset['soh'] > params.train_soh_bound
		train_mask = np.logical_and(train_mask, above_soh_mask)
	#endregion
	
	train_dict = {}
	test_dict = {}
	for pkey in dataset.keys():
		train_dict[pkey] = dataset[pkey][train_mask]
		test_dict[pkey] = dataset[pkey][test_mask]

	#region: normalize data if specified
	if params.normalize:
		# scale train data
		train_dict['input_scaler'] = StandardScaler().fit(train_dict[model_input_key])
		train_dict[f'{model_input_key}_scaled'] = train_dict['input_scaler'].transform(train_dict[model_input_key])
		train_dict['output_scaler'] = StandardScaler().fit(train_dict[model_output_key].reshape(-1,1))
		train_dict[f'{model_output_key}_scaled'] = train_dict['output_scaler'].transform(train_dict[model_output_key].reshape(-1,1))

		# scale test dataset (using scalers from training data)
		test_dict['input_scaler'] = train_dict['input_scaler']
		test_dict[f'{model_input_key}_scaled'] = test_dict['input_scaler'].transform(test_dict[model_input_key])
		test_dict['output_scaler'] = train_dict['output_scaler']
		test_dict[f'{model_output_key}_scaled'] = test_dict['output_scaler'].transform(test_dict[model_output_key].reshape(-1,1))

		# set model input and output keys
		train_dict['model_input'] = train_dict[f'{model_input_key}_scaled']
		train_dict['model_output'] = train_dict[f'{model_output_key}_scaled']
		test_dict['model_input'] = test_dict[f'{model_input_key}_scaled']
		test_dict['model_output'] = test_dict[f'{model_output_key}_scaled']
	else:
		# set model input and output keys
		train_dict['model_input'] = train_dict[model_input_key]
		train_dict['model_output'] = train_dict[model_output_key]
		test_dict['model_input'] = test_dict[model_input_key]
		test_dict['model_output'] = test_dict[model_output_key]
	#endregion
	
	return train_dict, test_dict

def train_model(model, X, y, epochs=200, n_batches=100, val_split=0.1):
	assert len(X) > 0, "The training set (Xt) contains no samples"

	#region: Define fit parameters
	batchsize = int(len(X) / n_batches)
	if batchsize < 1: batchsize = 1
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=False, mode='auto', baseline=None, restore_best_weights=True)
	#endregion

	history = model.fit(X, y, batch_size=batchsize, epochs=epochs, validation_split=val_split, callbacks=early_stop, verbose=0)
	
	return model, history

def fine_tune(model, Xt, yt, ft_params:FineTuning_Parameters, verbose=False, fit_params={}):
	''' 
	Model is finetuned using the provided target data \n
	model : keras model
	\tSource model (note that the model is passed by reference and therefore modified in place)
	Xt : dict
	\tInput features from target training data
	yt : dict
	\tOutputs from target training data
	ft_params : FineTuning_Parameters
	\tParameters specific to the fine-tuning process (see FineTuning_Parameters dataclass)
	verbose : bool, default False
	'''
	
	assert len(Xt) > 0, "The training set (Xt) contains no samples"

	if verbose:
		print("Performing fine-tuning...")
		print(f" - The model has {len(model.layers)} layers")
		print(f" - Only the last {ft_params.n_retrain_layers if ft_params.n_retrain_layers > 1 else ''} \
			  layer{'s' if ft_params.n_retrain_layers > 1 else ''} \
			  {'are' if ft_params.n_retrain_layers > 1 else 'is'} being trained")

	#region: Freeze specified layers
	n_frozen = len(model.layers) - ft_params.n_retrain_layers
	init_layer_weights = [model.layers[i].get_weights() for i in range(n_frozen)]
	for i in range(n_frozen):
		model.layers[i].trainable = False

	model.compile(optimizer=tf.keras.optimizers.Adam(ft_params.learning_rate_1), 
				  loss='mse', metrics=['mean_squared_error'], weighted_metrics=[])
	# endregion

	#region: Define fit parameters
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=False, mode='auto', baseline=None, restore_best_weights=True)
	#endregion

	ft1_history = model.fit(Xt, yt, batch_size=ft_params.batch_size_1, epochs=ft_params.epochs_1, validation_split=ft_params.val_split_1, callbacks=early_stop, verbose=0, **fit_params)
	
	#region: Check that frozen layers didn't change
	final_layer_weights = [model.layers[i].get_weights() for i in range(n_frozen)]
	for i in range(len(final_layer_weights)):
		for j in range(len(final_layer_weights[i])):
			np.testing.assert_allclose( init_layer_weights[i][j], final_layer_weights[i][j], err_msg="Weights changed in frozen layers. Check code for errors")
	#endregion

	#region: Unfreeze all layers and perform final fine-tuning
	for i in range(len(model.layers)):
		model.layers[i].trainable = True
	model.compile(optimizer=tf.keras.optimizers.Adam(ft_params.learning_rate_2), loss='mse', metrics=['mean_squared_error'], weighted_metrics=[])
	#endregion

	ft2_history = model.fit(Xt, yt, batch_size=ft_params.batch_size_2, epochs=ft_params.epochs_2, validation_split=ft_params.val_split_2, verbose=0, **fit_params)

	return model, ft1_history, ft2_history

def perform_transfer_learning(params:TransferLearning_Parameters, return_all:bool=False):
	"""_summary_

	Args:
		params (TransferLearning_Parameters): The full set of TL params used for this function call.
		return_all (bool, optional): Can optionally return the actual train and test data for both the source and target models, and the models themselves. Defaults to False.

	Returns:
		tuple: (params, error) if return_all is False. Else returns: (params, errors, source_train, source_test, target_train, target_test, source_model, ft_model, to_model)
	"""

	rand_seed = params.rand_seed
	import random
	random.seed(rand_seed)
	tf.random.set_seed(rand_seed)
	np.random.seed(rand_seed)

	print(f'  Seed: {rand_seed}\n', end='')

	errors = {
		f'{err_name}_{err_metric}':0 for err_name in fine_tuning_result_keys['model_error_names'].values()
									 for err_metric in fine_tuning_result_keys['error_metrics'].keys()
	}

	#region: get source and target datasets
	temp_data = get_processed_data(dataset_id=params.dataset_id, data_type='slowpulse')
	filt_idxs = np.where(temp_data['pulse_type'] == params.pulse_type)
	all_data = {
		'features':np.asarray([v - v[0] for v in temp_data['voltage'][filt_idxs]]),
		'targets':temp_data['soh'][filt_idxs],
		'cell_id':temp_data['cell_id'][filt_idxs],
		'group_id':temp_data['group_id'][filt_idxs],
		'soc':temp_data['soc'][filt_idxs],
		'soh':temp_data['soh'][filt_idxs],
	}
	# if source_target_split parameters are defined, we first want to split all data into two groups: 
	# one used for the source model and the other for the target model
	all_source_data = all_data
	all_target_data = all_data
	if params.source_target_split_params is not None:
		all_source_data, all_target_data = dataset_train_test_split(
			all_data, 
			params.source_target_split_params,
			rand_seed=rand_seed)

	#region filter source and target datasets to only the specified socs
	source_filt_idxs = None
	target_filt_idxs = None
	if hasattr(params.source_soc, '__len__'):
		for soc in params.source_soc:
			if source_filt_idxs is None:
				source_filt_idxs = (all_source_data['soc'] == soc)
			else:
				source_filt_idxs = np.logical_or(source_filt_idxs, (all_source_data['soc'] == soc))
	else:
		source_filt_idxs = (all_source_data['soc'] == params.source_soc)
	if hasattr(params.target_soc, '__len__'):
		for soc in params.target_soc:
			if target_filt_idxs is None:
				target_filt_idxs = (all_target_data['soc'] == soc)
			else:
				target_filt_idxs = np.logical_or(target_filt_idxs, (all_target_data['soc'] == soc))  
	else:
		target_filt_idxs = (all_target_data['soc'] == params.target_soc)
	#endregion
	
	source_dataset = {}
	target_dataset = {}
	for k,v in all_source_data.items():
		try:
			source_dataset[k] = v[source_filt_idxs]
		except TypeError: pass
	for k,v in all_target_data.items():
		try:
			target_dataset[k] = v[target_filt_idxs]
		except TypeError: 
			pass
	source_train, source_test = dataset_train_test_split(source_dataset, params.source_train_split_params, suppress_warning=True, rand_seed=rand_seed)
	target_train, target_test = dataset_train_test_split(target_dataset, params.target_train_split_params, suppress_warning=True, rand_seed=rand_seed)
	#endregion

	#region: build source, new, and fine-tuned models
	# create source model
	_temp = None
	if hasattr(params.source_soc, '__len__'): _temp = None
	else: _temp = params.source_soc
	try:
		mlp_params = get_optimal_model_params(
			model_to_use='mlp', 
			dataset_id=params.dataset_id, 
			pulse_type=params.pulse_type, 
			pulse_soc=_temp)
	except:
		print("Could not find optimal source model. Using default model size.")
		mlp_params = get_default_model_params(
			model_to_use='mlp', 
			dataset_id=params.dataset_id)
	params.n_hlayers = mlp_params['n_hlayers']
	params.n_neurons = mlp_params['n_neurons']

	model_source = create_MLP_model(
		**mlp_params, 
		input_shape=source_train['model_input'].shape[1], 
		output_shape=source_train['model_output'].shape[1])
	model_source, _ = train_model(model_source, source_train['model_input'], source_train['model_output'], epochs=500, n_batches=25)

	# perform fine tuning
	model_ft = create_MLP_model(
		**mlp_params, 
		input_shape=source_train['model_input'].shape[1], 
		output_shape=source_train['model_output'].shape[1])
	model_ft, _ = train_model(model_ft, source_train['model_input'], source_train['model_output'])

	# TODO: currently adding some source data into fine-tuning process to retain source accuracy
	num_source_samples = int(len(source_train['model_input']) * params.ft_params.percent_of_source_train_samples_to_use)
	rng = np.random.default_rng(seed=params.rand_seed)
	sample_idxs = rng.choice(np.arange(len(source_train['model_input'])), size=num_source_samples, axis=0, replace=False)
	ft_data_X = np.vstack([source_train['model_input'][sample_idxs], target_train['model_input']])  	# target_train['model_input']
	ft_data_y = np.vstack([source_train['model_output'][sample_idxs], target_train['model_output']])  	# target_train['model_output']
	# TODO: currently giving the target samples a different weight than the source samples
	sample_weight = np.hstack([
		np.full(shape=len(sample_idxs), fill_value=1), 
		np.full(shape=len(target_train['model_input']), fill_value=params.ft_params.target_to_source_sample_weight)])
	model_ft, _, _ = fine_tune(model_ft, ft_data_X, ft_data_y, params.ft_params, fit_params={'sample_weight':sample_weight})

	# create new model
	model_new = create_MLP_model(
		**mlp_params, 
		input_shape=source_train['model_input'].shape[1], 
		output_shape=source_train['model_output'].shape[1])
	model_new, _ = train_model(model_new, target_train['model_input'], target_train['model_output'])
	#endregion

	#region: perform predictions
	res = {}
	for err_name in fine_tuning_result_keys['model_error_names'].values():
		res[f'{err_name}_true'] = None 
		res[f'{err_name}_pred'] = None 
	
	yt_true = None
	if params.target_train_split_params.normalize:
		yt_true         = target_test['output_scaler'].inverse_transform( target_test['model_output'] )
		res['dt_pred']  = target_test['output_scaler'].inverse_transform( model_source.predict(target_test['model_input'], verbose=0) )
		res['to_pred']  = target_test['output_scaler'].inverse_transform( model_new.predict(target_test['model_input'], verbose=0) ) 
		res['ft_pred']  = target_test['output_scaler'].inverse_transform( model_ft.predict(target_test['model_input'], verbose=0) ) 
	else:
		yt_true         = target_test['model_output']
		res['dt_pred']  = model_source.predict(target_test['model_input'], verbose=0)
		res['to_pred']  = model_new.predict(target_test['model_input'], verbose=0)
		res['ft_pred']  = model_ft.predict(target_test['model_input'], verbose=0)

	if params.source_train_split_params.normalize:
		res['so_true']  = source_test['output_scaler'].inverse_transform( source_test['model_output'] )
		res['so_pred']  = source_test['output_scaler'].inverse_transform( model_source.predict(source_test['model_input'], verbose=0) )
	else:
		res['so_true']  = source_test['model_output']
		res['so_pred']  = model_source.predict(source_test['model_input'], verbose=0)
	res['dt_true'] = res['ft_true'] = res['to_true'] =  yt_true
	#endregion
		
	#region: Store Error Metrics
	for err_name in fine_tuning_result_keys['model_error_names'].values():
		for err_metric in fine_tuning_result_keys['error_metrics'].keys():
			ytrue = res[f'{err_name}_true']
			ypred = res[f'{err_name}_pred']
			errors[f'{err_name}_{err_metric}'] = fine_tuning_result_keys['error_metrics'][err_metric](ytrue, ypred)
	# endregion

	if return_all:
		misc = {
			'source_train':source_train,
			'source_test':source_test,
			'target_train':target_train,
			'target_test':target_test,
			
			'model_source':model_source,
			'model_ft':model_ft,
			'model_to':model_new,
		}
		return params, errors, misc
	return (params, errors)

def perform_transfer_learning_other_features(params:TransferLearning_Parameters, return_all:bool=False):
	"""_summary_

	Args:
		params (TransferLearning_Parameters): The full set of TL params used for this function call.
		return_all (bool, optional): Can optionally return the actual train and test data for both the source and target models, and the models themselves. Defaults to False.

	Returns:
		tuple: (params, error) if return_all is False. Else returns: (params, errors, source_train, source_test, target_train, target_test, source_model, ft_model, to_model)
	"""

	rand_seed = params.rand_seed
	print(f'  Seed: {rand_seed}\n', end='')

	errors = {
		f'{err_name}_{err_metric}':0 for err_name in fine_tuning_result_keys['model_error_names'].values()
									 for err_metric in fine_tuning_result_keys['error_metrics'].keys()
	}

	#region: get source and target datasets
	temp_data = get_processed_data(dataset_id=params.dataset_id, data_type='slowpulse')
	filt_idxs = np.where(temp_data['pulse_type'] == params.pulse_type)
	
	def _get_features(all_data, feature_id:str):
		assert feature_id in ['full_voltage', 'endpoints', 'peak', 'std', 'mean', 'mean_std', 'area', 'pca5']
		
		voltages = np.asarray([v - v[0] for v in all_data['voltage']])
		if feature_id == 'full_voltage':
			return voltages
		elif feature_id == 'endpoints':
			fs = np.vstack([
				voltages[:,1],  voltages[:,30],
				voltages[:,31], voltages[:,40],
				voltages[:,41], voltages[:,100]]).T
			assert len(fs) == len(voltages)
			return fs
		elif feature_id == 'peak':
			fs = voltages[:,40].reshape(-1,1)
			assert len(fs) == len(voltages)
			return fs
		elif feature_id == 'std':
			fs = np.std(voltages, axis=1).reshape(-1,1)
			assert len(fs) == len(voltages)
			return fs
		elif feature_id == 'mean':
			fs = np.mean(voltages, axis=1).reshape(-1,1)
			assert len(fs) == len(voltages)
			return fs
		elif feature_id == 'mean_std':
			fs = np.vstack([
				np.mean(voltages, axis=1),
				np.std(voltages, axis=1)]).T
			assert len(fs) == len(voltages)
			return fs
		elif feature_id == 'area':
			fs = np.sum(voltages, axis=1).reshape(-1,1)
			assert len(fs) == len(voltages)
			return fs
		elif feature_id == 'pca5':
			from sklearn.decomposition import PCA
			pca = PCA(n_components=5).fit(voltages)
			fs = pca.transform(voltages)
			assert len(fs) == len(voltages)
			return fs	

	features = _get_features(temp_data, params.feature_id)[filt_idxs]
	all_data = {
		'features':features,
		'targets':temp_data['soh'][filt_idxs],
		'cell_id':temp_data['cell_id'][filt_idxs],
		'group_id':temp_data['group_id'][filt_idxs],
		'soc':temp_data['soc'][filt_idxs],
		'soh':temp_data['soh'][filt_idxs],
	}
	# if source_target_split parameters are defined, we first want to split all data into two groups: 
	# one used for the source model and the other for the target model
	all_source_data = all_data
	all_target_data = all_data
	if params.source_target_split_params is not None:
		all_source_data, all_target_data = dataset_train_test_split(
			all_data, 
			params.source_target_split_params,
			rand_seed=rand_seed)

	#region filter source and target datasets to only the specified socs
	source_filt_idxs = None
	target_filt_idxs = None
	if hasattr(params.source_soc, '__len__'):
		for soc in params.source_soc:
			if source_filt_idxs is None:
				source_filt_idxs = (all_source_data['soc'] == soc)
			else:
				source_filt_idxs = np.logical_or(source_filt_idxs, (all_source_data['soc'] == soc))
	else:
		source_filt_idxs = (all_source_data['soc'] == params.source_soc)
	if hasattr(params.target_soc, '__len__'):
		for soc in params.target_soc:
			if target_filt_idxs is None:
				target_filt_idxs = (all_target_data['soc'] == soc)
			else:
				target_filt_idxs = np.logical_or(target_filt_idxs, (all_target_data['soc'] == soc))  
	else:
		target_filt_idxs = (all_target_data['soc'] == params.target_soc)
	#endregion
	
	source_dataset = {}
	target_dataset = {}
	for k,v in all_source_data.items():
		try:
			source_dataset[k] = v[source_filt_idxs]
		except TypeError: pass
	for k,v in all_target_data.items():
		try:
			target_dataset[k] = v[target_filt_idxs]
		except TypeError: 
			pass
	source_train, source_test = dataset_train_test_split(source_dataset, params.source_train_split_params, suppress_warning=True, rand_seed=rand_seed)
	target_train, target_test = dataset_train_test_split(target_dataset, params.target_train_split_params, suppress_warning=True, rand_seed=rand_seed)
	#endregion

	#region: build source, new, and fine-tuned models
	# create source model
	_temp = None
	if hasattr(params.source_soc, '__len__'): _temp = None
	else: _temp = params.source_soc
	
	try:
		mlp_params = get_optimal_model_params(
			model_to_use='mlp', 
			dataset_id=params.dataset_id, 
			pulse_type=params.pulse_type, 
			pulse_soc=_temp)
	except:
		mlp_params = get_default_model_params(
			model_to_use='mlp', 
			dataset_id=params.dataset_id)
	params.n_hlayers = mlp_params['n_hlayers']
	params.n_neurons = mlp_params['n_neurons']

	model_source = create_MLP_model(
		**mlp_params, 
		input_shape=source_train['model_input'].shape[1], 
		output_shape=source_train['model_output'].shape[1])
	model_source, _ = train_model(model_source, source_train['model_input'], source_train['model_output'], epochs=500, n_batches=25)

	# perform fine tuning
	model_ft = create_MLP_model(
		**mlp_params, 
		input_shape=source_train['model_input'].shape[1], 
		output_shape=source_train['model_output'].shape[1])
	model_ft, _ = train_model(model_ft, source_train['model_input'], source_train['model_output'])

	# TODO: currently adding some source data into fine-tuning process to retain source accuracy
	num_source_samples = int(len(source_train['model_input']) * params.ft_params.percent_of_source_train_samples_to_use)
	rng = np.random.default_rng(seed=params.rand_seed)
	sample_idxs = rng.choice(np.arange(len(source_train['model_input'])), size=num_source_samples, axis=0, replace=False)
	ft_data_X = np.vstack([source_train['model_input'][sample_idxs], target_train['model_input']])  	# target_train['model_input']
	ft_data_y = np.vstack([source_train['model_output'][sample_idxs], target_train['model_output']])  	# target_train['model_output']
	# TODO: currently giving the target samples a different weight than the source samples
	sample_weight = np.hstack([
		np.full(shape=len(sample_idxs), fill_value=1), 
		np.full(shape=len(target_train['model_input']), fill_value=params.ft_params.target_to_source_sample_weight)])
	model_ft, _, _ = fine_tune(model_ft, ft_data_X, ft_data_y, params.ft_params, fit_params={'sample_weight':sample_weight})

	# create new model
	model_new = create_MLP_model(
		**mlp_params, 
		input_shape=source_train['model_input'].shape[1], 
		output_shape=source_train['model_output'].shape[1])
	model_new, _ = train_model(model_new, target_train['model_input'], target_train['model_output'])
	#endregion

	#region: perform predictions
	res = {}
	for err_name in fine_tuning_result_keys['model_error_names'].values():
		res[f'{err_name}_true'] = None 
		res[f'{err_name}_pred'] = None 
	
	yt_true = None
	if params.target_train_split_params.normalize:
		yt_true         = target_test['output_scaler'].inverse_transform( target_test['model_output'] )
		res['dt_pred']  = target_test['output_scaler'].inverse_transform( model_source.predict(target_test['model_input'], verbose=0) )
		res['to_pred']  = target_test['output_scaler'].inverse_transform( model_new.predict(target_test['model_input'], verbose=0) ) 
		res['ft_pred']  = target_test['output_scaler'].inverse_transform( model_ft.predict(target_test['model_input'], verbose=0) ) 
	else:
		yt_true         = target_test['model_output']
		res['dt_pred']  = model_source.predict(target_test['model_input'], verbose=0)
		res['to_pred']  = model_new.predict(target_test['model_input'], verbose=0)
		res['ft_pred']  = model_ft.predict(target_test['model_input'], verbose=0)

	if params.source_train_split_params.normalize:
		res['so_true']  = source_test['output_scaler'].inverse_transform( source_test['model_output'] )
		res['so_pred']  = source_test['output_scaler'].inverse_transform( model_source.predict(source_test['model_input'], verbose=0) )
	else:
		res['so_true']  = source_test['model_output']
		res['so_pred']  = model_source.predict(source_test['model_input'], verbose=0)
	res['dt_true'] = res['ft_true'] = res['to_true'] =  yt_true
	#endregion
		
	#region: Store Error Metrics
	for err_name in fine_tuning_result_keys['model_error_names'].values():
		for err_metric in fine_tuning_result_keys['error_metrics'].keys():
			ytrue = res[f'{err_name}_true']
			ypred = res[f'{err_name}_pred']
			errors[f'{err_name}_{err_metric}'] = fine_tuning_result_keys['error_metrics'][err_metric](ytrue, ypred)
	# endregion

	if return_all:
		return params, errors, source_train, source_test, target_train, target_test, model_source, model_ft, model_new
	return (params, errors)


def check_params_in_df(params:TransferLearning_Parameters, df:pd.DataFrame):
	new_row = params.as_dataframe()
	new_row.drop(columns=['rand_seed'], inplace=True)
	cols_to_compare = new_row.columns.values

	if not float(new_row['source_target_train_size'].values[0]) == 0.0:
		cols_to_compare = [key for key in cols_to_compare if key not in ['source_target_train_ids', 'source_target_test_ids']]
	if not float(new_row['source_train_size'].values[0]) == 0.0:
		cols_to_compare = [key for key in cols_to_compare if key not in ['source_train_ids', 'source_test_ids']]
	if not float(new_row['target_train_size'].values[0]) == 0.0:
		cols_to_compare = [key for key in cols_to_compare if key not in ['target_train_ids', 'target_test_ids']]

	if (None in df['n_hlayers'].unique()) or (params.n_hlayers is None):
		cols_to_compare = [key for key in cols_to_compare if key not in ['n_hlayers']]
	if (None in df['n_neurons'].unique()) or (params.n_neurons is None):
		cols_to_compare = [key for key in cols_to_compare if key not in ['n_neurons']]

	# Old tests don't have a feature_id so drop it from comparison
	if not 'feature_id' in df.columns:
		cols_to_compare = [key for key in cols_to_compare if key not in ['feature_id']]

	prior_df = df[cols_to_compare].astype(str)
	new_row = new_row[cols_to_compare].astype(str)

	prior_size = prior_df.drop_duplicates().shape[0]
	post_df = pd.concat([prior_df, new_row])
	post_size = post_df.drop_duplicates().shape[0]

	# if the size is the same, then the new parameters already exist in the df
	return prior_size == post_size

def multiprocess_transfer_learning(n_iterations:int, param_generator, test_name:str, f_results:Path, overwrite_existing:bool=False, max_processes:int=5):
	'''
	Runs \'perform_transfer_learning\' over several iterations with the average and std of the resulting errors recorded
	n_iterations : int
	\tThe number of iterations to run for each parameter set
	param_generator : generator obbject
	\tEach iteration of the generator should return a single TransferLearning_Parameters object
	test_name : str
	\tA name for test being run. Used for naming the saved results
	f_results : Path
	\tPath object of a folder for where the results should be stored
	overwrite_existing : bool, default False
	\tIf True, all existing files will be overwritten with new results. Otherwise the existing tests will be skipped
	max_processes : int, default 5
	\tThe maximum number of processes that can run concurrently 

	Returns: path to saved pickle file containg the results dataframe
	'''

	f_test_results = f_results.joinpath(test_name)
	f_test_results.mkdir(parents=False, exist_ok=True)
	filename_suffix = 0
	filename_results = f"{test_name}_Results_{filename_suffix}.pkl"
	
	for count, param_iter in enumerate(param_generator):
		print(f"Iteration {count}: {repr(param_iter)}\n", end='')
		
		#region: Load previous results (if exist)
		df_prev_results = None
		if not overwrite_existing:
			param_already_tested = False
			for f in list(f_test_results.glob('*.pkl')):
				# for every existing file, check if this parameter was already tested
				df_prev_results = pickle.load(open(f, 'rb'))
				if check_params_in_df(param_iter, df_prev_results): 
					print("  Parameter set already exists in saved results ... skipping")
					param_already_tested = True
					break
				# update file index to match current file
				start_idx = f.name.rindex(f"{test_name}_Results_")+len(f"{test_name}_Results_")
				end_idx = f.name.rindex('.pkl')
				filename_suffix = max(filename_suffix, int(f.name[start_idx:end_idx]))
			
			# jump to next iteration if already tested
			if param_already_tested: continue

		temp_filepath = f_test_results.joinpath(f"{test_name}_Results_{filename_suffix}.pkl")
		if temp_filepath.exists():
			df_prev_results = pickle.load(open(temp_filepath, 'rb'))
		else:
			df_prev_results = pd.DataFrame()
		#endregion
		
		#region: perform multiprocessing on repeated iterations (limited to 5 processes)
		# create a repeated set of param_iter with new random_seed
		mp_params = []
		for i in range(n_iterations):
			param = deepcopy(param_iter)
			param.rand_seed = i
			mp_params.append(param)
	
		# create pool of processes
		pool = mp.Pool(processes=np.min([max_processes, n_iterations]))
		pool_res = None
		if test_name == 'Other_Features':
			print(f"  Feature: {param_iter.feature_id}")
			pool_res = pool.map(perform_transfer_learning_other_features, mp_params)
		else:
			pool_res = pool.map(perform_transfer_learning, mp_params)
		pool.close()
		pool.join()
		#endregion

		#region: get all return values from each process that was run (record all iterations to be averaged later)
		all_errors = {}
		for err_key in pool_res[0][1].keys():
			all_vals = [pool_res[i][1][err_key] for i in range(len(pool_res))]
			all_errors[f'{err_key}'] = all_vals
		#endregion

		# Combine errors and parameters into a single dataframe
		df_errors = pd.DataFrame({k:v for k,v in all_errors.items()})
		df_params = pd.DataFrame()
		for params in [pool_res[i][0] for i in range(len(pool_res))]:
			if len(df_params) == 0: 
				df_params = params.as_dataframe()
			else: 
				df_params = pd.concat([df_params, params.as_dataframe()], ignore_index=True)
		assert len(df_params) == len(df_errors)
		df_combined = pd.concat([df_params, df_errors], axis=1)

		# set column names if no previous results exist
		if df_prev_results.empty:
			df_prev_results = df_prev_results.reindex(df_prev_results.columns.union( df_combined.columns ), axis=1)

		# concatenate current iteration results to previous results and save dataframe
		df_prev_results = pd.concat([df_prev_results, df_combined], axis=0, ignore_index=True)
		df_prev_results = convert_TLasDF_datatypes(df_prev_results)
		filename_results = f"{test_name}_Results_{filename_suffix}.pkl"
		pickle.dump(df_prev_results, open(f_test_results.joinpath(filename_results), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
		if len(df_prev_results) >= 5000: filename_suffix += 1        # start saving to new file if dataframe becomes too large
		
		# delete dataframe variables to improve memory consumptions
		del df_prev_results
		del df_combined

def get_splitting_params_for_single_group(dataset_id:str, group_id:int, rand_seed:int):
	'''Return the (source_split, target_split) parameters for a single random split of the provided cycling group (split by cells)'''

	assert dataset_id in get_available_dataset_ids()
	
	rng = np.random.default_rng(seed=rand_seed)
	all_cells = get_cell_ids_in_group(dataset_id, group_id)

	#region: select source and target cells for train/test
	source_train = source_test = target_train = target_test = None
	if len(all_cells) == 4:
		source_train, source_test = rng.choice(all_cells, (2,1), replace=False)
		target_train, target_test = rng.choice([i for i in all_cells if i not in source_train and i not in source_test], 
											(2,1), replace=False)
	else:
		source_train = rng.choice(all_cells, 2, replace=False)
		source_test = rng.choice([i for i in all_cells if i not in source_train], 1, replace=False)
		target_train = rng.choice([i for i in all_cells if i not in source_train and i not in source_test], 
								2, replace=False)
		target_test = rng.choice([i for i in all_cells if i not in source_train and i not in source_test and i not in target_train], 
								1, replace=False)
	#endregion

	# define transfer learning and splitting parameters
	source_split = Splitting_Parameters(
		split_unit='cell', train_ids=source_train, test_ids=source_test, normalize=True
	)
	target_split = Splitting_Parameters(
		split_unit='cell', train_ids=target_train, test_ids=target_test, normalize=True
	)

	return source_split, target_split

def group_iter_already_tested(params_to_check:dict, df:pd.DataFrame) -> bool:
	"""_summary_

	Args:
		params_to_check (dict): _description_. Must contain the following keys: ['source_group_id', 'target_group_id']
		df (pd.DataFrame): _description_

	Returns:
		bool: Returns True if params_to_check exist in df
	"""

	assert 'source_group_id' in list(params_to_check.keys())
	assert 'target_group_id' in list(params_to_check.keys())

	# mask df to matching params (for only the parameters than exist in both params_to_check and df.columns)
	shared_cols = [x for x in list(params_to_check.keys()) if x in df.columns.to_list()]
	mask = np.full(1, len(df))
	for i in range(len(shared_cols)):
		col_mask = (df[shared_cols[i]].astype(str) == str(params_to_check[shared_cols[i]])).values
		mask = np.logical_and(mask, col_mask)

	# convert the df source/target cell ids to source/target group ids
	source_train_cells = [str_list_to_list(x) for x in df['source_train_ids'][mask].values]
	source_test_cells = [str_list_to_list(x) for x in df['source_test_ids'][mask].values]
	source_groups = [get_group_ids_from_cells(df['chemistry'][mask].iloc[i], 
										  np.hstack([source_train_cells[i], source_test_cells[i]])) 
					for i in range(len(source_train_cells))]
	source_groups = np.ravel(np.asarray(source_groups))

	target_train_cells = [str_list_to_list(x) for x in df['target_train_ids'][mask].values]
	target_test_cells = [str_list_to_list(x) for x in df['target_test_ids'][mask].values]
	target_groups = [get_group_ids_from_cells(df['chemistry'][mask].iloc[i], 
										  np.hstack([target_train_cells[i], target_test_cells[i]])) 
					for i in range(len(target_train_cells))]
	target_groups = np.ravel(np.asarray(target_groups))

	# creat mask where source and target group ids match params_to_check
	group_mask = np.logical_and(source_groups == params_to_check['source_group_id'], target_groups == params_to_check['target_group_id'])

	# return True if this parameter set exists in df
	if len(df[mask][group_mask]) > 0:
		return True
	
	return False

class DistributionDifference:
	def __init__(self, A:np.ndarray, B:np.ndarray, nbins:int=100):
		if not isinstance(A, np.ndarray): raise TypeError(f"A must be a numpy array. {type(A)} != {np.ndarray}")
		if not isinstance(B, np.ndarray): raise TypeError(f"B must be a numpy array. {type(B)} != {np.ndarray}")
		# assert A.shape == B.shape, f"A and B must have the same shape. {A.shape != B.shape}"
		assert len(A.shape) > 0 and len(A.shape) < 3, "DistributionDifference currently only supports 1-dimensional or 2-dimensional data"

		# initiallize two distributions and calculate their probability density functions
		self._A = A
		self._B = B
		self._nbins = nbins
		all_data = np.hstack([A.flatten(), B.flatten()])
		self._bins = np.linspace(np.min(all_data), np.max(all_data), num=self._nbins, endpoint=True)

		#region: create pdfs of A
		self.ps = []
		if len(self._A.shape) == 1:
			self.ps.append(self._calculate_pdfs(self._A))
		else:
			self.ps = []
			for i in range(self._A.shape[1]):
				self.ps.append(self._calculate_pdfs(self._A[:,i]))
		self.ps = np.asarray(self.ps)
		#endregion

		#region: create pdfs of B
		self.qs = []
		if len(self._B.shape) == 1:
			self.qs.append(self._calculate_pdfs(self._B))
		else:
			self.qs = []
			for i in range(self._B.shape[1]):
				self.qs.append(self._calculate_pdfs(self._B[:,i]))
		self.qs = np.asarray(self.qs)
		#endregion

		#region: calculate difference metrics
		self._KLD = self._calculate_KLD()
		self._cross_entropy = self._calculate_cross_entropy()
		self._wasserstein = self._calculate_earth_movers()
		self._itakura_saito = self._calculate_itakura_saito_distance()
		self._maximum_mean_discrepancy = self._calculate_maximum_mean_discrepancy()
		#endregion
		
	def _calculate_pdfs(self, p):
		if not (p.sum() > 0.999 and p.sum() < 1.001) or not (len(p) == len(self._bins)-1):
			counts, _ = np.histogram(p, bins=self._bins)
			a = counts / counts.sum()
		else: a = p
		assert a.sum() > 0.999 and a.sum() < 1.001
		return a
				
	def _calculate_KLD(self, epsilon=1e-6):
		"""KLD = np.sum( p * np.log(p/q) )"""
		total = 0
		for i in range(len(self.ps)):
			total += np.sum((self.ps[i]+epsilon) * np.log((self.ps[i]+epsilon) / (self.qs[i]+epsilon)))
		return total

	def _calculate_cross_entropy(self, epsilon=1e-6):
		"""Entropy = -1 * np.sum( p * np.log(q) )"""
		total = 0
		for i in range(len(self.ps)):
			total += -np.sum( (self.ps[i] + epsilon) * np.log((self.qs[i] + epsilon)) )
		return total

	def _calculate_earth_movers(self):
		"""Earth Movers (Wasserstein) distance"""
		from scipy.stats import wasserstein_distance
		total = 0
		for i in range(len(self.ps)):
			total += wasserstein_distance(self.ps[i], self.qs[i])
		return total

	def _calculate_itakura_saito_distance(self, epsilon=1e-6):
		"""Itakura-Saito distance"""
		total = 0
		for i in range(len(self.ps)):
			total += np.sum( (self.ps[i] + epsilon)/(self.qs[i] + epsilon) - np.log((self.ps[i] + epsilon)/(self.qs[i] + epsilon)) - 1 )
		return total
	
	def _calculate_maximum_mean_discrepancy(self, gamma=1.0):
		"""Maximum Mean Discrepancy (MMD)"""
		from sklearn.metrics.pairwise import rbf_kernel
		K_AA = rbf_kernel(self.ps, self.ps, gamma)
		K_AB = rbf_kernel(self.ps, self.qs, gamma)
		K_BB = rbf_kernel(self.qs, self.qs, gamma)
		return np.mean(K_AA) + np.mean(K_BB) - (2 * np.mean(K_AB))

	@property
	def kullback_leibler(self):
		return self._KLD
	@property
	def cross_entropy(self):
		return self._cross_entropy
	@property
	def wasserstein(self):
		return self._wasserstein
	@property
	def itakura_saito(self):
		return self._itakura_saito
	@property
	def maximum_mean_discrepancy(self):
		return self._maximum_mean_discrepancy

	def as_dict(self):
		return {
			'kullback_leibler': self._KLD,
			'kld': self._KLD,
			'cross_entropy': self._cross_entropy,
			'entropy': self._cross_entropy,
			'wasserstein':self._wasserstein,
			'emd':self._wasserstein,
			'itakura_saito': self._itakura_saito,
			'isd': self._itakura_saito,
			'maximum_mean_discrepancy': self._maximum_mean_discrepancy,
			'mmd': self._maximum_mean_discrepancy,
		}

	def plot(self, **kwargs):
		pass



#region: plotting functions
def get_unit_str_from_error_metric(error_metric:str, test_variable_unit:Optional[str]=None):
	assert error_metric in ['rmse', 'mae', 'mape', 'r2']
	if error_metric in ['rmse', 'mae']: 
		assert test_variable_unit is not None, "The specified error metrics has units equal to the test variable. Please provide a value for \'test_variable_unit\'."
	unit_dic = {
		'rmse':test_variable_unit,
		'mae':test_variable_unit,
		'mape':'%',
		'r2':'-'
	}
	return unit_dic[error_metric]

def plot_finetuning_optimization_results(dataset_id:str, pulse_type:str, test_name:str, test_variable:str, error_metric:str, x_label:str=None, plot_all_socs=False, mark_optimal:bool=False, save_folder=None):
	"""_summary_

	Args:
		dataset_id (str): _description_
		pulse_type (str): _description_
		test_name (str): _description_
		test_variable (str): _description_
		error_metric (str): _description_
		x_label (str, optional): _description_. Defaults to None.
		plot_all_socs (bool, optional): _description_. Defaults to False.
		mark_optimal (bool, optional): _description_. Defaults to False.
		save_folder (_type_, optional): _description_. Defaults to None.
	"""

	assert dataset_id in ['UConn-ILCC-LFP', 'UConn-ILCC-NMC']
	assert dir_results.joinpath('finetuning_optimization', "st_split_by_cell", test_name).is_dir(), f"Cannot locate directory: {dir_results.joinpath('finetuning_optimization', 'st_split_by_cell', test_name)}"
	assert error_metric in ['rmse', 'mae', 'mape', 'r2']
	assert pulse_type in ['chg', 'dchg']
	
	#region: load test results
	dfs = []
	for file_res in dir_results.joinpath('finetuning_optimization', "st_split_by_cell", test_name).glob('*.pkl'):
		dfs.append(pickle.load(open(file_res, 'rb')))
	df = pd.concat(dfs, ignore_index=True)
	df = df.loc[(df['dataset_id'] == dataset_id) & (df['pulse_type'] == pulse_type)]
	source_soc_range = df['source_soc'].unique()
	target_soc_range = df['target_soc'].unique()
	#endregion

	chemistry = dic_available_dataset_info[dataset_id]['cell_chemistry']

	results = {
		'test_variable':[],
		'source':[],
		'direct_transfer':[],
		'target_only':[],
		'fine_tuning':[],
	}

	result_key_mapping = {
		'source':fine_tuning_result_keys['model_error_names']['source_on_source'], 
		'direct_transfer':fine_tuning_result_keys['model_error_names']['source_on_target'], 
		'target_only':fine_tuning_result_keys['model_error_names']['target_only'],
		'fine_tuning':fine_tuning_result_keys['model_error_names']['fine_tuning']}
	plot_label_mapping = {'source':'Source', 'direct_transfer':'Direct', 'target_only':'Target-Only', 'fine_tuning':'Fine-Tuning'}
	for source_soc in source_soc_range:
		for target_soc in target_soc_range:
			if source_soc == target_soc: continue
			df_filt = df.loc[(df['source_soc'] == source_soc) & (df['target_soc'] == target_soc)]
			test_variable_range = np.sort(df_filt[test_variable].unique())
			cur_results = {
				'test_variable':test_variable_range,
				'source':np.zeros((2, len(test_variable_range))),
				'direct_transfer':np.zeros((2, len(test_variable_range))),
				'target_only':np.zeros((2, len(test_variable_range))),
				'fine_tuning':np.zeros((2, len(test_variable_range))),
			}
		
			# each test variable value has multiple repeated tests -> take average
			error_keys = [f'{k}_{error_metric}' for k in list(result_key_mapping.values())]
			error_keys.append(test_variable)
			error_keys = [k for k in error_keys if k in (df_filt.columns)]
			df_avg = df_filt[error_keys].groupby(test_variable).mean().sort_values(test_variable)
			df_std = df_filt[error_keys].groupby(test_variable).std().sort_values(test_variable)

			for k in [k for k in list(cur_results.keys()) if not k == 'test_variable']:
				cur_results[k][0] = df_avg[f'{result_key_mapping[k]}_{error_metric}'].values * (100 if error_metric == 'mape' else 1)
				cur_results[k][1] = df_std[f'{result_key_mapping[k]}_{error_metric}'].values * (100 if error_metric == 'mape' else 1)

			# append results at current SOC combination to all other results (for averaging later on)
			for k in cur_results.keys():
				results[k].append( cur_results[k] )

			#region: plot each results at each SOC
			if plot_all_socs:
				fig, ax = plt.subplots(figsize=(4,2.5))
				for i, k in enumerate([k for k in list(cur_results.keys()) if (not k == 'test_variable') and (not k == 'source')]):
					ax.plot(cur_results['test_variable'], cur_results[k][0], label=plot_label_mapping[k], c=f'C{i}')
					ax.errorbar(cur_results['test_variable'], cur_results[k][0], 
								yerr=cur_results[k][1], fmt='o', capsize=2, color=f'C{i}')
					if mark_optimal:
						optimal_y = np.min(cur_results[k][0])
						optimal_x = cur_results['test_variable'][np.argmin(cur_results[k][0])]
						ax.plot(optimal_x, optimal_y, 's', markersize=10, c=f'C{i}',
								markerfacecolor='none', markeredgecolor=f'C{i}', linewidth=1.0)
						
				ax.legend(fontsize=8, ncols=3, loc='upper right')
				ax.set_xlabel(x_label)
				if 'learning_rate' in test_variable:
					ax.set_xscale('log')
				ax.set_ylabel(f"{error_metric.upper()}{' [%]' if error_metric == 'mape' else ''}")
				ax.set_title(f'{chemistry} ({pulse_type}): {source_soc}% to {target_soc}%')
				fig.tight_layout(pad=0.8)
				if save_folder is not None:
					assert isinstance(save_folder, Path)
					save_path = save_folder.joinpath(
						f"OptimizationTests_{test_name}",
						f"OptimizationTests_{test_name}_{chemistry}_{pulse_type}_{source_soc}-{target_soc}{'_showOptimal' if mark_optimal else ''}.png")
					save_path.parent.mkdir(parents=True, exist_ok=True)
					plt.savefig(save_path, dpi=300)
				plt.show()
			#endregion

	# average all individual SOC combinations
	min_array_size = np.min([len(x) for k in results.keys() 
						  			for x in results[k]])
	for k in results.keys():
		#region: need to filter results for FineTuningLayers test 
			# Each soc combination can have different model size, thus a different number of layers retrained
			# For averaging, we need all arrays to have the same length
			# The arrays will be shortened to the smallest array length available
			# i.e., only n_retrain_layers between 1-3 will be plotted in the smallest model used in 3 layers
		filt_results = None
		if len(results[k][0].shape) > 1:
			filt_results = np.asarray([x[:, :min_array_size] for x in results[k]])
		else:
			filt_results = np.asarray([x[:min_array_size] for x in results[k]])
		#endregion
		if test_name == 'FineTuningLayers': results[k] = filt_results
		else: results[k] = np.asarray(results[k])
		if k == 'test_variable':
			results[k] = np.average(results[k], axis=0)
		else:
			results[k] = [np.average(results[k][:,0], axis=0),
						np.average(results[k][:,1], axis=0)]
	
	#region: plot averages across all SOC combinations
	if not plot_all_socs:
		fig, ax = plt.subplots(figsize=(4,2.5))
		for i, k in enumerate([k for k in list(results.keys()) if (not k == 'test_variable') and (not k == 'source')]):
			ax.plot(results['test_variable'], results[k][0], label=plot_label_mapping[k], c=f'C{i}')
			ax.errorbar(results['test_variable'], results[k][0], 
						yerr=results[k][1], fmt='o', capsize=2, color=f'C{i}')
			if mark_optimal:
				optimal_y = np.min(results[k][0])
				optimal_x = results['test_variable'][np.argmin(results[k][0])]
				ax.plot(optimal_x, optimal_y, 's', markersize=10, c=f'C{i}',
						markerfacecolor='none', markeredgecolor=f'C{i}', linewidth=1.0)
			
		ax.legend(fontsize=8, ncols=3, loc='upper right')
		ax.set_xlabel(x_label)
		if 'learning_rate' in test_variable:
			ax.set_xscale('log')
		ax.set_ylabel(f"{error_metric.upper()}{' [%]' if error_metric == 'mape' else ''}")
		ax.set_title(f'{chemistry} ({pulse_type}): Average of All SOCs')
		fig.tight_layout(pad=0.8)
		if save_folder is not None:
			assert isinstance(save_folder, Path)
			save_path = save_folder.joinpath(
				f"OptimizationTests_{test_name}",
				f"OptimizationTests_{test_name}_{chemistry}_{pulse_type}_AverageOfAllSOCs{'_showOptimal' if mark_optimal else ''}.png")
			save_path.parent.mkdir(parents=True, exist_ok=True)
			plt.savefig(save_path, dpi=300)
		plt.show()
	#endregion
	
	return results


class ErrorFunction:
	def __init__(self, fnc:Any, repr:str="ErrorFunction"):
		self._fnc = fnc
		self._repr = repr

	def __call__(self, A:Union[float, np.ndarray], B:Union[float, np.ndarray]):
		if not type(A) == type(B): raise TypeError(f"A and B must be the same type. {type(A)} != {type(B)}")
		if isinstance(A, np.ndarray):
			res = np.vectorize(self._fnc)(A, B)
		else:
			res = self._fnc(A, B)
		return res
	
	def __repr__(self):
		return self._repr
PercentChange = ErrorFunction(
	fnc = lambda x,y: (x - y) / (1 if y == 0 else y) * 100,
	repr = "Calculates the percent change in value between A and B: (A - B) / B * 100"
)
AbsoluteChange = ErrorFunction(
	fnc = lambda x,y: (x - y), 
	repr = "Calculates the absolute change in value between A and B: (A - B)"
)

def get_soc_error_results(dataset_id:str, pulse_type:str, results_to_show:str, error_metric:str, return_type:str='mean', remove_outliers:bool=False, split_by:str='cell') -> np.ndarray:
	"""Returns the SOC-combination results for the specified dataset and model (source-only, finetuning, target-only)

	Args:
		dataset_id (str): _description_
		pulse_type (str): {'chg', 'dchg'}
		results_to_show (str): {'source_on_source', 'source_on_target', 'target_only', 'fine_tuning'}
		error_metric (str): {'rmse', 'mae', 'mape', 'r2'}
		return_type (str, optional): {'mean', 'std', 'values'}. What to return for each SOC combination (e.g., the mean of values, the std, the raw values, et). Defaults to 'mean'.
		remove_outliers (bool, optional): Will drop any outliers before taking the mean or standard deviation. Outliers are only dropped if 'return_type' is not equal to 'values' Defaults to False

	Returns:
		np.ndarray: A 2-dimensional array of the results. Use [target_soc][source_soc] to access individual results -> eg. results[0][1] = 20% SOC to 10% SOC
	"""

	assert dataset_id in ['UConn-ILCC-LFP', 'UConn-ILCC-NMC']
	assert pulse_type in ['chg', 'dchg']
	assert results_to_show in ['source_on_source', 'source_on_target', 'target_only', 'fine_tuning']
	assert error_metric in ['rmse', 'mae', 'mape', 'r2']
	assert return_type in ['mean', 'std', 'values']

	f_saved_results = dir_results.joinpath("finetuning_optimization", f"st_split_by_{split_by}", "FixedParameters")
	assert f_saved_results.is_dir(), f"Cannot find directory: {f_saved_results}"

	dfs = [] 
	for file in f_saved_results.glob('*.pkl'):
		dfs.append(pickle.load(open(file, 'rb')))
	df = pd.concat(dfs, ignore_index=True)

	#region: get errors corresponding to 'results_to_show'
	df_filt = df.loc[(df['dataset_id'] == dataset_id) & (df['pulse_type'] == pulse_type)]
	source_socs = sorted(df_filt['source_soc'].astype(int).unique())
	target_socs = sorted(df_filt['target_soc'].astype(int).unique())
	results = np.zeros(shape=(len(target_socs), len(source_socs)))
	if return_type == 'values':
		results = []
	
	for i, target_soc in enumerate(target_socs):
		if return_type == 'values': results.append([])
		for j, source_soc in enumerate(source_socs):
			if source_soc == target_soc: 
				if return_type == 'values': results[-1].append([])
				continue
			df_soc = df_filt.loc[(df_filt['source_soc'].astype(int) == source_soc) & (df_filt['target_soc'].astype(int) == target_soc)]

			#region: remove outliers in any of the error metric columns if specified
			if remove_outliers and not (return_type == 'values'):
				err_cols = [f'source_{error_metric}', f'dt_{error_metric}', f'new_{error_metric}', f'ft_{error_metric}']
				err_cols = [err for err in err_cols if err in list(df_soc.columns)]
				from scipy import stats
				df_soc = df_soc[(np.abs(stats.zscore(df_soc[err_cols])) < 3.0).all(axis=1)]
			#endregion

			error_key = f"{fine_tuning_result_keys['model_error_names'][results_to_show]}_{error_metric}"
			values = df_soc[error_key].values * (100 if error_metric == 'mape' else 1)

			if return_type == 'mean': results[i,j] = np.mean(values)
			elif return_type == 'std': results[i,j] = np.std(values)
			elif return_type == 'values': results[-1].append(values)            
	#endregion
	
	#region: if returning raw values, we need to replace diagonal with array of zeros
	if return_type == 'values':
		num_vals = 0
		for i in range(len(results)):
			for j in range(len(results[i])):
				if len(results[i][j]) > num_vals: num_vals = len(results[i][j])
		for i in range(len(results)):
			results[i][i] = np.zeros(shape=num_vals)
		results = np.asarray(results)
	#endregion
	
	return np.asarray(results)

def reject_outliers(data:np.ndarray, m:float=2.0, keep_dims:bool=False) -> tuple:
	"""_summary_

	Args:
		data (np.ndarray): _description_
		m (float, optional): _description_. Defaults to 2.0.
		keep_dims (bool, optional): _description_. Defaults to False.

	Returns:
		tuple: (array, count). Array is the clean data. Count is number of outliers detected.
	"""

	# calculate absolute dist from median for all values
	d = np.sort(np.abs(data - np.median(data)))
	# caculate the median distance 
	mdev = np.median(d)
	# create a sudo-standard deviation metric using the median 
	s = d/mdev if mdev else np.zeros(len(d))
	# only return data less than the specified number of median-based standard deviations
	idxs_no_outliers = s<m
	idxs_outliers = s>=m
	ret = deepcopy(data)
	if keep_dims:
		ret[idxs_outliers] = np.median(data)
		return ret, np.count_nonzero(idxs_outliers) > 0
	else:
		return ret[idxs_no_outliers], np.count_nonzero(idxs_outliers)

def plot_soc_combination_grid(results:np.ndarray, annotations:np.ndarray=None, **kwargs) -> tuple:
	"""Plots the provided matrix of results in a standard format

	Args:
		results (np.ndarray): A 2-dimensional array of results. Rows are plotted on the y-axis, columns on the x-axis. 
		annotations (np.ndarray, optional): If not None, annotations are added to the center of each grid with the annotation value/text corresponding to the same value in the annotations array
		**kwargs: Available kwargs include: ['fig', 'axes', 'cmap', 'norm', 'error_bounds', 'figsize', 'title', 'xlabel', 'ylabel', 'xticklabels', 'yticklabels', 'cbar_label', 'cbar_nticks', 'annotation_fontsize']

	Returns:
		tuple: (fig, axes, cbar). The figure instances are returned
	"""
	assert len(results.shape) == 2

	#region: set kwargs
	fig, axes = None, None
	if 'fig' in kwargs and 'axes' in kwargs:
		fig = kwargs['fig']
		axes = kwargs['axes']
	elif 'fig' in kwargs: assert 'axes' in kwargs, "Both 'fig' and 'axes' needed to be provided."
	elif 'axes' in kwargs: assert 'fig' in kwargs, "Both 'fig' and 'axes' needed to be provided."

	cmap = mpl.cm.Blues
	if 'cmap' in kwargs: cmap = kwargs['cmap']
	
	error_bounds = (np.min(results), np.max(results))
	if 'error_bounds' in kwargs: 
		error_bounds = kwargs['error_bounds']
	
	error_pad = 0.02 * abs(np.diff(error_bounds))		# add 2% buffer on bounds
	norm = mpl.colors.Normalize(vmin=error_bounds[0]-error_pad, vmax=error_bounds[1]+error_pad)
	if 'norm' in kwargs: norm = kwargs['norm']

	figsize = (3.25, 2.5)
	if 'figsize' in kwargs: figsize = kwargs['figsize']

	annotation_fontsize = 8
	if 'annotation_fontsize' in kwargs: annotation_fontsize = kwargs['annotation_fontsize']
	use_adaptive_color = True
	#endregion

	#region: set up figure
	scm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
	if fig is None or axes is None:
		fig = plt.figure(figsize=figsize, constrained_layout=True)
		gs = GridSpec(1, 2, figure=fig, width_ratios=[15,1])
		axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
	#endregion

	# plot results
	axes[0].imshow(results, cmap=cmap, norm=norm, aspect='auto', origin='lower', interpolation='nearest')

	#region: add annotation on each square of heatmap
	if annotations is not None:
		assert annotations.shape[0:2] == results.shape[0:2], "annotations must have the same first two dimensions as results"
		
		#region: check whether only diagonals are nonzero
		only_diags = False
		has_diags = True

		diag_idxs = np.diag_indices(annotations.shape[0])
		triu_idxs = np.triu_indices(annotations.shape[0], k=1)
		tril_idxs = np.tril_indices(annotations.shape[0], k=-1)
		try:
			if np.all([x is None for x in annotations[diag_idxs]]) or np.all(annotations[diag_idxs].astype(int) == 0):
				only_diags = False
				has_diags = False
			elif (np.all([x is None for x in annotations[triu_idxs]]) or np.all(annotations[triu_idxs].astype(int) == 0)) and \
				(np.all([x is None for x in annotations[tril_idxs]]) or np.all(annotations[tril_idxs].astype(int) == 0)) :
				only_diags = True
		except ValueError:
			only_diags = False
		#endregion
		
		for (j,i), label in np.ndenumerate(annotations):
			if i == j and not has_diags: continue
			if only_diags and not(i == j): continue
			
			# convert label to int if decimal is 0
			if isinstance(label, int) or isinstance(label, float):
				if label == np.floor(label): label = int(label)
			# add results value to center of square
			axes[0].text(i,j, label, ha='center', va='center', fontsize=annotation_fontsize,
				color='white' if use_adaptive_color and results[j,i] > (0.75*(error_bounds[1]-error_bounds[0]) + error_bounds[0]) else 'black')
	#endregion

	#region: set plot title, labels, and ticks if specified in kwargs
	if 'title' in kwargs:
		axes[0].set_title(kwargs['title'])
	if 'xlabel' in kwargs:
		axes[0].set_xlabel(kwargs['xlabel'])
	if 'ylabel' in kwargs:
		axes[0].set_ylabel(kwargs['ylabel'])
	if 'xticklabels' in kwargs:
		axes[0].set_xticks(np.arange(0, len(kwargs['xticklabels']), 1), labels=kwargs['xticklabels'])
	if 'yticklabels' in kwargs:
		axes[0].set_yticks(np.arange(0, len(kwargs['yticklabels']), 1), labels=kwargs['yticklabels'])
	cbar_label = ''
	if 'cbar_label' in kwargs: cbar_label = kwargs['cbar_label']
	cbar_nticks = 5
	if 'cbar_nticks' in kwargs: cbar_nticks = kwargs['cbar_nticks']
	#endregion
	
	# add colorbar
	extend_lower = np.min(results) < error_bounds[0]
	extend_upper = np.max(results) > error_bounds[1]
	extend_key = 'neither'
	if extend_lower and extend_upper: extend_key = 'both'
	elif extend_lower: extend_key = 'min'
	elif extend_upper: extend_key = 'max'
	cbar = fig.colorbar(scm, cax=axes[1], label=cbar_label, extend=extend_key)
	cbar.set_ticks(np.linspace(error_bounds[0], error_bounds[1], num=cbar_nticks, endpoint=True))

	return fig, axes, cbar

def get_finetuning_errors(dataset_id:str, pulse_type:str, results_to_compare:tuple=('source_on_source', 'source_on_target'),
	error_metric:str='mape', error_difference_fnc:ErrorFunction=PercentChange, remove_outliers:bool=False, split_by:str='cell'):
	"""_summary_

	Args:
		dataset_id (str): {'UConn-ILCC-LFP', 'UConn-ILCC-NMC'}
		pulse_type (str): {'chg', 'dchg'}
		results_to_compare (tuple, optional): {'source_on_source', 'source_on_target', 'fine_tuning', 'target_only'}. Defaults to ('source_on_source', 'source_on_target').
		error_metric (str, optional): {'mape', 'mae', 'rmse', 'r2'}. Defaults to 'mape'.
		error_difference_fnc (ErrorFunction, optional): A ErrorFunction class to calculate the difference between results. Defaults to PercentChange.
		remove_outliers (bool, optional): Can optionally specify whether to remove outliers before calculating the difference in results. Defaults to False.

	Returns:
		dict: {'mean', 'std', 'pass_count', 'total_count', 'params'}. The mean differences, the standard deviations, the number of times results_to_compare[0] perform better than results_to_compare[1], and the total number of values being evaluted per difference. Params holds the specific parameters being evaluated.
	"""

	# get errors for each result type
	results_A = get_soc_error_results(
		dataset_id=dataset_id,
		pulse_type=pulse_type,
		results_to_show=results_to_compare[0],
		error_metric=error_metric,
		return_type='values',
		split_by=split_by)
	results_B = get_soc_error_results(
		dataset_id=dataset_id,
		pulse_type=pulse_type,
		results_to_show=results_to_compare[1],
		error_metric=error_metric,
		return_type='values',
		split_by=split_by)
	
	# count how many times results_A errors are better than results_B
	pass_count = np.count_nonzero(results_A < results_B, axis=2)
	total_count = np.count_nonzero(results_A, axis=2)
	if error_metric == 'r2': pass_count = np.count_nonzero(results_A > results_B, axis=2)  # for r2, larger is better
	
	# calculate error difference with specified function
	results = error_difference_fnc(results_A, results_B)
	
	# if specified, remove any outliers from results
	outlier_count = np.zeros(shape=results.shape[:2], dtype=object)
	if remove_outliers:
		for i in range(results.shape[0]):
			for j in range(results.shape[1]):
				results[i][j], c = reject_outliers(results[i][j], m=3, keep_dims=True)
				outlier_count[i][j] = c

	# get result statistics
	means = np.mean(results, axis=2)
	stds = np.std(results, axis=2)

	params = {
		'dataset_id':dataset_id, 'pulse_type':pulse_type, 'results_to_compare':results_to_compare, 
		'error_metric':error_metric, 'error_difference_fnc':error_difference_fnc, 
		'remove_outliers':remove_outliers, 'outlier_count':outlier_count
	}
	ret_dict = {
		'mean':means, 'std':stds, 'pass_count':pass_count, 'total_count':total_count, 'params':params
	}
	return ret_dict

def plot_finetuning_errors(results:dict, annotation_type:Union[str, tuple]='mean', **kwargs):
	"""_summary_

	Args:
		results (dict): A dict containing the following keys: {'mean', 'std', 'pass_count', 'total_count', 'params'}. Use the dict returned from 'get_finetuning_errors()'
		annotation_type (Union[str, tuple], optional): One or a list of the following: {'mean', 'std', 'pass', 'pass/total', 'pass/fail'}. The value(s) to annotate. Defaults to 'mean'.
		**kwargs: keyword arguments to be used in 'plot_soc_coombination_grid()'
	Returns:
		tuple: (fig, ax, cbar). Return the figure instances
	"""

	assert isinstance(results, dict)
	for k in ['mean', 'std', 'pass_count', 'total_count', 'params']: 
		assert k in results.keys(), "results must contain the following keys: ['mean', 'std', 'pass_count', 'total_count', 'params']"
	if isinstance(annotation_type, str): annotation_type = [annotation_type]
	for k in annotation_type: assert k in ['mean', 'std', 'pass', 'pass/total', 'pass/fail']

	#region: create annotations
	annotation_decimals = 0
	max_val = np.max(abs(results['mean']))
	if 'error_bounds' in kwargs: 
		max_val = np.max(np.hstack([np.abs(kwargs['error_bounds'][0]), np.abs(kwargs['error_bounds'][1]), max_val]))
	if max_val < 0: annotation_decimals = 3
	elif max_val < 2: annotation_decimals = 2
	elif max_val < 10: annotation_decimals = 1

	annotations = np.empty(shape=results['pass_count'].shape, dtype=object)
	for i in range(results['pass_count'].shape[0]): 
		for j in range(results['pass_count'].shape[1]):
			if annotations[i][j] is None: annotations[i][j] = ''		# initiallize annotation str
			# if i == j: continue											# don't annotate diagonal
			for k, ann_type in enumerate(annotation_type):							# add each specified annotation type as a new line
				has_outlier = results['params']['outlier_count'][i][j] > 0
				nl = '\n'
				if ann_type == 'mean':
					annotations[i][j] += f"{nl if k > 0 else ''}{np.round(results['mean'][i][j], annotation_decimals).astype(int if annotation_decimals == 0 else float)}{'*' if (has_outlier and k==0) else ''}"
				elif ann_type == 'std':
					annotations[i][j] += f"{nl if k > 0 else ''}{np.round(results['std'][i][j], annotation_decimals).astype(int if annotation_decimals == 0 else float)}{'*' if (has_outlier and k==0) else ''}"
				elif ann_type == 'pass':
					annotations[i][j] += f"{nl if k > 0 else ''}{results['pass_count'][i][j]}{'*' if (has_outlier and k==0) else ''}"
				elif ann_type == 'pass/total':
					annotations[i][j] += f"{nl if k > 0 else ''}{results['pass_count'][i][j]}/{results['total_count'][i][j]}{'*' if (has_outlier and k==0) else ''}"
				elif ann_type == 'pass/fail':
					annotations[i][j] += f"{nl if k > 0 else ''}{results['pass_count'][i][j]}/{results['total_count'][i][j] - results['pass_count'][i][j]}{'*' if (has_outlier and k==0) else ''}"
				else: raise ValueError(f"Annotation type not supported ({ann_type})")
	#endregion

	#region: set title
	comparison_name_A = fine_tuning_result_keys['model_error_names'][results['params']['results_to_compare'][0]].upper()
	comparison_name_B = fine_tuning_result_keys['model_error_names'][results['params']['results_to_compare'][1]].upper()
	title = f"{comparison_name_A} w.r.t. {comparison_name_B}"
	if 'title' in kwargs:
		title = kwargs.pop('title')
	#endregion

	soc_bol = r"$SOC_{BOL}$"
	# plot soc grid
	soc_range = dic_available_dataset_info[results['params']['dataset_id']]['pulse_socs_tested']
	fig, ax, cbar = plot_soc_combination_grid(
		results=results['mean'], 
		annotations=annotations,
		xlabel=f"Source {soc_bol} [%]", 
		ylabel=f"Target {soc_bol} [%]", 
		xticklabels=soc_range, 
		yticklabels=soc_range,
		title=title,
		**kwargs)
	
	return fig, ax, cbar
#endregion








# Each key in the below 'test_definitions' dict creates a new optimization study.
test_definitions = {
	# 'FineTuningLayers':{	
	# 	'n_iterations': 10,
	# 	'test_values': [1, 2, 3, 4, 5],
	# },
	# 'FineTuningLearningRate1':{
	# 	'n_iterations': 10,   
	# 	'test_values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
	# },
	# 'FineTuningLearningRate2':{
	# 	'n_iterations': 10,
	# 	'test_values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
	# },
	# 'FineTuningEpochs2':{
	# 	'n_iterations': 10,
	# 	'test_values': [5, 10, 15, 20, 25, 30],
	# },
	# 'TargetTrainSize':{
	# 	'n_iterations': 10,
	# 	'test_values': [1, 2, 3, 4, 5, 6, 7, 8],
	# },
	# 'FixedParameters':{
	#     'n_iterations': 100,
	#     'test_values': None,
	# },
	# 'MultiSourceSOC_to_SingleTargetSOC_2t1':{
	#     'n_iterations': 20,
	#     'test_values': {
	# 		'UConn-ILCC-LFP':[((20,50),90), ((20,90),50), ((50,90),20)],
	# 		'UConn-ILCC-NMC':[
	# 			((30,40),10), ((30,40),20), ((30,40),50), ((30,40),60), ((30,40),70), ((30,40),80), ((30,40),90),
	# 			((30,50),10), ((30,50),20), ((30,50),40), ((30,50),60), ((30,50),70), ((30,50),80), ((30,50),90),
	# 			((30,60),10), ((30,60),20), ((30,60),40), ((30,60),50), ((30,60),70), ((30,60),80), ((30,60),90),
	# 			((40,50),10), ((40,50),20), ((40,50),30), ((40,50),60), ((40,50),70), ((40,50),80), ((40,50),90),
	# 			((10,90),20), ((10,90),30), ((10,90),40), ((10,90),50), ((10,90),60), ((10,90),70), ((10,90),80),
	# 		]
	# 	},
	# },
	# 'Other_Features':{
	# 	'n_iterations':10,
	# 	'test_values': ['full_voltage', 'endpoints', 'peak', 'std', 'mean', 'mean_std', 'area', 'pca5']
	# 	# 'test_values': ['endpoints', 'peak', 'std', 'mean', 'area', 'pca5']
	# }
}

def TransferLearning_Parameter_Generator(test_name:str, source_target_split:str) -> Generator[TransferLearning_Parameters]:
	"""_summary_

	Args:
		test_name (str): 
		source_target_split (str): {'none', 'cell', 'group'}.
			'cell' will split the full dataset into source and target datasets, each with unique cell ids.
			'group' will split the full dataset into source and target datasets, each with unique group ids.
			'none' will use the full dataset for both source and target (the same cell ids will appear in both source and target datasets).
	Yields:
		TransferLearning_Parameters: _description_
	"""
	
	assert source_target_split in ['none', 'cell', 'group']
	assert test_name in list(test_definitions.keys())

	split_params_source_target = None
	if source_target_split == 'cell':
		split_params_source_target = Splitting_Parameters(
			split_unit      = 'cell',
			split_method    = 'percent',
			train_size      = 0.666,        # use 66% of all cells for source model
			test_size       = 0.333,        # use remaining cells for target model
			stratify        = True,
			normalize       = False )
	elif source_target_split == 'group':
		split_params_source_target = Splitting_Parameters(
			split_unit      = 'group', 
			split_method    = 'count', 
			train_size      = 8,            # use 8 groups for source model
			test_size       = 3,            # use remaining 3 groups for target model
			stratify        = False, 
			normalize       = False)


	def _get_uniform_test_size(dataset_id, total_cell_count, split_unit, split_method, train_size):
		'''returns the maximum test size (in the same units at split_units)'''
		
		assert dataset_id in get_available_dataset_ids()
		assert split_unit in ['cell', 'group']
		assert split_method in ['count', 'percent']

		if split_unit == 'cell':
			if split_method == 'count':
				return total_cell_count - int(train_size)
			else:   # percent
				return 1 - train_size

		elif split_unit == 'group':
			cells_per_group = (6 if dataset_id == 'UConn-ILCC-LFP' else 4)
			total_groups = int(total_cell_count / cells_per_group)
			if split_method == 'count':
				assert total_groups > train_size, \
					f"Number of groups used for training ({train_size}) is greater than or equal to the total number of groups ({total_groups})"
				return int(total_groups - int(train_size))
			else:   # percent
				return 1 - train_size
	
	for dataset_id in ['UConn-ILCC-LFP', 'UConn-ILCC-NMC']:
		soc_range = dic_available_dataset_info[dataset_id]['pulse_socs_tested']
		for pulse_type in ['chg', 'dchg']:
			#region: define target size, we need to know the number of cells in source and target datasets
			num_cells = dic_available_dataset_info[dataset_id]['cell_count']
			num_source_cells = num_cells
			num_target_cells = num_cells
			if split_params_source_target is not None:
				if split_params_source_target.split_unit == 'cell':
					num_source_cells = None
					if split_params_source_target.split_method == 'count':
						num_source_cells = int(split_params_source_target.train_size)
					elif split_params_source_target.split_method == 'percent':
						num_source_cells = round(split_params_source_target.train_size * num_cells)
				elif split_params_source_target.split_unit == 'group':
					cells_per_group = (6 if dataset_id == 'UConn-ILCC-LFP' else 4)
					if split_params_source_target.split_method == 'count':
						num_source_cells = int(split_params_source_target.train_size) * cells_per_group
					elif split_params_source_target.split_method == 'percent':
						num_source_cells = round(split_params_source_target.train_size * 11) * cells_per_group
				num_target_cells = num_cells - num_source_cells
			#endregion
			
			#region: define multiSourceSOC tests
			if test_name == 'MultiSourceSOC_to_SingleTargetSOC_2t1':
				for soc_groups in test_definitions[test_name]['test_values'][dataset_id]:
					source_socs = soc_groups[0]
					target_soc = soc_groups[1]
					
					# define source splitting parameters
					split_params_source = Splitting_Parameters(
						split_unit      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_unit'],
						split_method    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_method'], 
						train_size      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_size'],
						test_size       = _get_uniform_test_size(
							dataset_id			= dataset_id, 
							total_cell_count    = num_source_cells, 
							split_unit          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_unit'],
							split_method        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_method'],
							train_size          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_size'] ),
						train_soh_bound = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_soh_bound'],
						stratify        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_stratify'],
						normalize       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_normalize'] )
				
					# define target splitting parameters
					split_params_target = Splitting_Parameters(
						split_unit      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_unit'],
						split_method    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_method'], 
						train_size      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_size'],
						test_size       = _get_uniform_test_size(
							dataset_id			= dataset_id, 
							total_cell_count    = num_target_cells, 
							split_unit          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_unit'],
							split_method        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_method'],
							train_size          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_size'] ),
						train_soh_bound = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_soh_bound'],
						stratify        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_stratify'],
						normalize       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_normalize'] )

					# define fine_tuning parameters
					ft_params = FineTuning_Parameters(
						n_retrain_layers    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_n_retrain_layers'],
						learning_rate_1     = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_learning_rate_1'],
						learning_rate_2     = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_learning_rate_2'],
						epochs_1            = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_epochs_1'],
						epochs_2            = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_epochs_2'],
						batch_size_1        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_batch_size_1'],
						batch_size_2        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_batch_size_2'],
						val_split_1         = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_val_split_1'],
						val_split_2         = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_val_split_2'],)

					# define transfer_learning parameters
					tl_params = TransferLearning_Parameters(
						dataset_id      = dataset_id, 
						pulse_type      = pulse_type, 
						source_soc      = source_socs, 
						target_soc      = target_soc, 
						# n_neurons       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['n_neurons'], 
						# n_hlayers       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['n_hlayers'], 
						source_target_split_params  = split_params_source_target,
						source_train_split_params   = split_params_source, 
						target_train_split_params   = split_params_target, 
						ft_params                   = ft_params )
				
					yield tl_params
			#endregion

			#region: single SOC tests:
			else:
				for source_soc in soc_range:
					for target_soc in soc_range:
						if source_soc == target_soc: continue

						if test_name == 'FixedParameters':
							# define source splitting parameters
							split_params_source = Splitting_Parameters(
								split_unit      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_unit'],
								split_method    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_method'], 
								train_size      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_source_cells, 
									split_unit          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_unit'],
									split_method        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_method'],
									train_size          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_size'] ),
								train_soh_bound = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_soh_bound'],
								stratify        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_stratify'],
								normalize       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_normalize'] )
							
							# define target splitting parameters
							split_params_target = Splitting_Parameters(
								split_unit      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_unit'],
								split_method    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_method'], 
								train_size      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_target_cells, 
									split_unit          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_unit'],
									split_method        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_method'],
									train_size          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_size'] ),
								train_soh_bound = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_soh_bound'],
								stratify        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_stratify'],
								normalize       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_normalize'] )

							# define fine_tuning parameters
							ft_params = FineTuning_Parameters(
								n_retrain_layers    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_n_retrain_layers'],
								learning_rate_1     = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_learning_rate_1'],
								learning_rate_2     = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_learning_rate_2'],
								epochs_1            = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_epochs_1'],
								epochs_2            = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_epochs_2'],
								batch_size_1        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_batch_size_1'],
								batch_size_2        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_batch_size_2'],
								val_split_1         = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_val_split_1'],
								val_split_2         = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_val_split_2'],)

							# define transfer_learning parameters
							tl_params = TransferLearning_Parameters(
								dataset_id      = dataset_id, 
								pulse_type      = pulse_type, 
								source_soc      = source_soc, 
								target_soc      = target_soc, 
								# n_neurons       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['n_neurons'], 
								# n_hlayers       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['n_hlayers'], 
								source_target_split_params  = split_params_source_target,
								source_train_split_params   = split_params_source, 
								target_train_split_params   = split_params_target, 
								ft_params                   = ft_params )
							
							yield tl_params

						elif test_name == 'FineTuningLayers':
							# define source splitting parameters
							split_params_source = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['source_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['source_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['source_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_source_cells, 
									split_unit          = DEFAULT_TL_PARAMS['source_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['source_split_method'],
									train_size          = DEFAULT_TL_PARAMS['source_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['source_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['source_stratify'],
								normalize       = DEFAULT_TL_PARAMS['source_normalize'] )
							
							# define target splitting parameters
							split_params_target = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['target_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['target_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['target_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_target_cells, 
									split_unit          = DEFAULT_TL_PARAMS['target_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['target_split_method'],
									train_size          = DEFAULT_TL_PARAMS['target_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['target_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['target_stratify'],
								normalize       = DEFAULT_TL_PARAMS['target_normalize'] )

							for n_ft_layers in test_definitions[test_name]['test_values']:
								model_hlayers = get_optimal_model_params(
									model_to_use='mlp', 
									dataset_id=dataset_id, 
									pulse_type=pulse_type, 
									pulse_soc=source_soc)['n_hlayers']
								if n_ft_layers > model_hlayers: continue
								# define fine_tuning parameters
								ft_params = FineTuning_Parameters(
									n_retrain_layers    = n_ft_layers,
									learning_rate_1     = DEFAULT_TL_PARAMS['ft_learning_rate_1'],
									learning_rate_2     = DEFAULT_TL_PARAMS['ft_learning_rate_2'],
									epochs_1            = DEFAULT_TL_PARAMS['ft_epochs_1'],
									epochs_2            = DEFAULT_TL_PARAMS['ft_epochs_2'],
									batch_size_1        = DEFAULT_TL_PARAMS['ft_batch_size_1'],
									batch_size_2        = DEFAULT_TL_PARAMS['ft_batch_size_2'],
									val_split_1         = DEFAULT_TL_PARAMS['ft_val_split_1'],
									val_split_2         = DEFAULT_TL_PARAMS['ft_val_split_2'],)

								# define transfer_learning parameters
								tl_params = TransferLearning_Parameters(
									dataset_id      = dataset_id, 
									pulse_type      = pulse_type, 
									source_soc      = source_soc, 
									target_soc      = target_soc, 
									source_target_split_params  = split_params_source_target,
									source_train_split_params   = split_params_source, 
									target_train_split_params   = split_params_target, 
									ft_params                   = ft_params)
								
								yield tl_params

						elif test_name == 'FineTuningLearningRate1':
							# define source splitting parameters
							split_params_source = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['source_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['source_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['source_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_source_cells, 
									split_unit          = DEFAULT_TL_PARAMS['source_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['source_split_method'],
									train_size          = DEFAULT_TL_PARAMS['source_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['source_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['source_stratify'],
								normalize       = DEFAULT_TL_PARAMS['source_normalize'] )
							
							# define target splitting parameters
							split_params_target = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['target_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['target_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['target_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_target_cells, 
									split_unit          = DEFAULT_TL_PARAMS['target_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['target_split_method'],
									train_size          = DEFAULT_TL_PARAMS['target_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['target_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['target_stratify'],
								normalize       = DEFAULT_TL_PARAMS['target_normalize'] )

							
							for lr1 in test_definitions[test_name]['test_values']:
								# define fine_tuning parameters
								ft_params = FineTuning_Parameters(
									n_retrain_layers    = DEFAULT_TL_PARAMS['ft_n_retrain_layers'],
									learning_rate_1     = lr1,
									learning_rate_2     = DEFAULT_TL_PARAMS['ft_learning_rate_2'],
									epochs_1            = DEFAULT_TL_PARAMS['ft_epochs_1'],
									epochs_2            = DEFAULT_TL_PARAMS['ft_epochs_2'],
									batch_size_1        = DEFAULT_TL_PARAMS['ft_batch_size_1'],
									batch_size_2        = DEFAULT_TL_PARAMS['ft_batch_size_2'],
									val_split_1         = DEFAULT_TL_PARAMS['ft_val_split_1'],
									val_split_2         = DEFAULT_TL_PARAMS['ft_val_split_2'],)
							
								# define transfer_learning parameters
								tl_params = TransferLearning_Parameters(
									dataset_id      = dataset_id, 
									pulse_type      = pulse_type, 
									source_soc      = source_soc, 
									target_soc      = target_soc, 
									source_target_split_params  = split_params_source_target,
									source_train_split_params   = split_params_source, 
									target_train_split_params   = split_params_target, 
									ft_params                   = ft_params )
								
								yield tl_params

						elif test_name == 'FineTuningLearningRate2':
							# define source splitting parameters
							split_params_source = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['source_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['source_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['source_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_source_cells, 
									split_unit          = DEFAULT_TL_PARAMS['source_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['source_split_method'],
									train_size          = DEFAULT_TL_PARAMS['source_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['source_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['source_stratify'],
								normalize       = DEFAULT_TL_PARAMS['source_normalize'] )
							
							# define target splitting parameters
							split_params_target = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['target_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['target_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['target_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_target_cells, 
									split_unit          = DEFAULT_TL_PARAMS['target_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['target_split_method'],
									train_size          = DEFAULT_TL_PARAMS['target_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['target_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['target_stratify'],
								normalize       = DEFAULT_TL_PARAMS['target_normalize'] )

							
							for lr2 in test_definitions[test_name]['test_values']:
								# define fine_tuning parameters
								ft_params = FineTuning_Parameters(
									n_retrain_layers    = DEFAULT_TL_PARAMS['ft_n_retrain_layers'],
									learning_rate_1     = DEFAULT_TL_PARAMS['ft_learning_rate_1'],
									learning_rate_2     = lr2,
									epochs_1            = DEFAULT_TL_PARAMS['ft_epochs_1'],
									epochs_2            = DEFAULT_TL_PARAMS['ft_epochs_2'],
									batch_size_1        = DEFAULT_TL_PARAMS['ft_batch_size_1'],
									batch_size_2        = DEFAULT_TL_PARAMS['ft_batch_size_2'],
									val_split_1         = DEFAULT_TL_PARAMS['ft_val_split_1'],
									val_split_2         = DEFAULT_TL_PARAMS['ft_val_split_2'],)
							
								# define transfer_learning parameters
								tl_params = TransferLearning_Parameters(
									dataset_id      = dataset_id, 
									pulse_type      = pulse_type, 
									source_soc      = source_soc, 
									target_soc      = target_soc, 
									source_target_split_params  = split_params_source_target,
									source_train_split_params   = split_params_source, 
									target_train_split_params   = split_params_target, 
									ft_params                   = ft_params )
								
								yield tl_params

						elif test_name == 'FineTuningEpochs2':
							# define source splitting parameters
							split_params_source = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['source_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['source_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['source_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_source_cells, 
									split_unit          = DEFAULT_TL_PARAMS['source_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['source_split_method'],
									train_size          = DEFAULT_TL_PARAMS['source_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['source_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['source_stratify'],
								normalize       = DEFAULT_TL_PARAMS['source_normalize'] )
							
							# define target splitting parameters
							split_params_target = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['target_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['target_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['target_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_target_cells, 
									split_unit          = DEFAULT_TL_PARAMS['target_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['target_split_method'],
									train_size          = DEFAULT_TL_PARAMS['target_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['target_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['target_stratify'],
								normalize       = DEFAULT_TL_PARAMS['target_normalize'] )

							
							for epoch2 in test_definitions[test_name]['test_values']:
								# define fine_tuning parameters
								ft_params = FineTuning_Parameters(
									n_retrain_layers    = DEFAULT_TL_PARAMS['ft_n_retrain_layers'],
									learning_rate_1     = DEFAULT_TL_PARAMS['ft_learning_rate_1'],
									learning_rate_2     = DEFAULT_TL_PARAMS['ft_learning_rate_2'],
									epochs_1            = DEFAULT_TL_PARAMS['ft_epochs_1'],
									epochs_2            = epoch2,
									batch_size_1        = DEFAULT_TL_PARAMS['ft_batch_size_1'],
									batch_size_2        = DEFAULT_TL_PARAMS['ft_batch_size_2'],
									val_split_1         = DEFAULT_TL_PARAMS['ft_val_split_1'],
									val_split_2         = DEFAULT_TL_PARAMS['ft_val_split_2'],)
							
								# define transfer_learning parameters
								tl_params = TransferLearning_Parameters(
									dataset_id      = dataset_id, 
									pulse_type      = pulse_type, 
									source_soc      = source_soc, 
									target_soc      = target_soc, 
									source_target_split_params  = split_params_source_target,
									source_train_split_params   = split_params_source, 
									target_train_split_params   = split_params_target, 
									ft_params                   = ft_params )
								
								yield tl_params

						elif test_name == 'TargetTrainSize':
							# define source splitting parameters
							split_params_source = Splitting_Parameters(
								split_unit      = DEFAULT_TL_PARAMS['source_split_unit'],
								split_method    = DEFAULT_TL_PARAMS['source_split_method'], 
								train_size      = DEFAULT_TL_PARAMS['source_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_source_cells, 
									split_unit          = DEFAULT_TL_PARAMS['source_split_unit'],
									split_method        = DEFAULT_TL_PARAMS['source_split_method'],
									train_size          = DEFAULT_TL_PARAMS['source_train_size'] ),
								train_soh_bound = DEFAULT_TL_PARAMS['source_train_soh_bound'],
								stratify        = DEFAULT_TL_PARAMS['source_stratify'],
								normalize       = DEFAULT_TL_PARAMS['source_normalize'] )

							# define fine_tuning parameters
							ft_params = FineTuning_Parameters(
								n_retrain_layers    = DEFAULT_TL_PARAMS['ft_n_retrain_layers'],
								learning_rate_1     = DEFAULT_TL_PARAMS['ft_learning_rate_1'],
								learning_rate_2     = DEFAULT_TL_PARAMS['ft_learning_rate_2'],
								epochs_1            = DEFAULT_TL_PARAMS['ft_epochs_1'],
								epochs_2            = DEFAULT_TL_PARAMS['ft_epochs_2'],
								batch_size_1        = DEFAULT_TL_PARAMS['ft_batch_size_1'],
								batch_size_2        = DEFAULT_TL_PARAMS['ft_batch_size_2'],
								val_split_1         = DEFAULT_TL_PARAMS['ft_val_split_1'],
								val_split_2         = DEFAULT_TL_PARAMS['ft_val_split_2'],)
							

							for n_target_train_cells in test_definitions[test_name]['test_values']:
								#define target splitting parameters
								split_params_target = Splitting_Parameters(
									split_unit      = 'cell',
									split_method    = 'count', 
									train_size      = n_target_train_cells,
									test_size       = _get_uniform_test_size(
										dataset_id			= dataset_id, 
										total_cell_count    = num_target_cells, 
										split_unit          = 'cell',
										split_method        = 'count',
										train_size          = max(test_definitions[test_name]['test_values']) ),
									train_soh_bound = DEFAULT_TL_PARAMS['target_train_soh_bound'],
									stratify        = DEFAULT_TL_PARAMS['target_stratify'],
									normalize       = DEFAULT_TL_PARAMS['target_normalize'] )

								# define transfer_learning parameters
								tl_params = TransferLearning_Parameters(
									dataset_id      = dataset_id, 
									pulse_type      = pulse_type, 
									source_soc      = source_soc, 
									target_soc      = target_soc,
									source_target_split_params  = split_params_source_target,
									source_train_split_params   = split_params_source, 
									target_train_split_params   = split_params_target, 
									ft_params                   = ft_params )
							
								yield tl_params

						elif test_name == 'Other_Features':
							# define source splitting parameters
							split_params_source = Splitting_Parameters(
								split_unit      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_unit'],
								split_method    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_method'], 
								train_size      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_source_cells, 
									split_unit          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_unit'],
									split_method        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_split_method'],
									train_size          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_size'] ),
								train_soh_bound = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_train_soh_bound'],
								stratify        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_stratify'],
								normalize       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['source_normalize'] )
							
							# define target splitting parameters
							split_params_target = Splitting_Parameters(
								split_unit      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_unit'],
								split_method    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_method'], 
								train_size      = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_size'],
								test_size       = _get_uniform_test_size(
									dataset_id			= dataset_id, 
									total_cell_count    = num_target_cells, 
									split_unit          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_unit'],
									split_method        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_split_method'],
									train_size          = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_size'] ),
								train_soh_bound = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_train_soh_bound'],
								stratify        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_stratify'],
								normalize       = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['target_normalize'] )

							# define fine_tuning parameters
							ft_params = FineTuning_Parameters(
								n_retrain_layers    = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_n_retrain_layers'],
								learning_rate_1     = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_learning_rate_1'],
								learning_rate_2     = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_learning_rate_2'],
								epochs_1            = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_epochs_1'],
								epochs_2            = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_epochs_2'],
								batch_size_1        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_batch_size_1'],
								batch_size_2        = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_batch_size_2'],
								val_split_1         = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_val_split_1'],
								val_split_2         = OPTIMAL_TL_PARAMS[dataset_id][pulse_type]['ft_val_split_2'],)

							for feature_id in test_definitions[test_name]['test_values']:
								# define transfer_learning parameters
								tl_params = TransferLearning_Parameters(
									dataset_id      = dataset_id, 
									pulse_type      = pulse_type, 
									source_soc      = source_soc, 
									target_soc      = target_soc, 
									source_target_split_params  = split_params_source_target,
									source_train_split_params   = split_params_source, 
									target_train_split_params   = split_params_target, 
									ft_params                   = ft_params,
									feature_id=feature_id)
						
								yield tl_params

						
						else:
							raise ValueError(f"\'{test_name}\' has not been defined. Please add the testing parameters to the generator function")
			#endregion




if __name__ == '__main__':
	# ============================= Record Transfer Learning Results for Defined Tests =======================================
	
	# Run all tests, using random_cell_selection (unique cell ids in source and target datasets)
	print("Starting Random Cell Selection Tests (split source and target into unique cell ids)")
	for test_name in test_definitions.keys():
		print(f"\n\nStarting Test: {test_name}")

		split_by = 'cell' # ['cell', 'group']

		results_folder = dir_results.joinpath("finetuning_optimization", f"st_split_by_{split_by}")
		results_folder.mkdir(exist_ok=True, parents=True)
		multiprocess_transfer_learning(
			n_iterations        = test_definitions[test_name]['n_iterations'],
			param_generator     = TransferLearning_Parameter_Generator(test_name, split_by),
			test_name           = test_name,
			f_results           = results_folder,
			overwrite_existing  = False,
			max_processes       = 5,
		)
		print(f"Test Complete: {test_name}\n")
		
	print()
	print("fine_tuning.py() complete")