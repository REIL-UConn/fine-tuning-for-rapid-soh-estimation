

import sys, os, warnings, re, pickle
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from sklearn.decomposition import PCA


from scripts.config import dic_available_dataset_info, dir_spreadsheets, dir_processed_data


def interp_time_series(ts:np.ndarray, ys:np.ndarray, n_points:int) -> tuple:
	"""Interpolates all y arrays to n_points based on a shared time array

	Args:
		ts (np.ndarray): An array of time values corresponding to every entry in ys
		ys (np.ndarray): A single array (or several stacked arrays) of values corresponding to ts
		n_points (int): The output length of ts and ys

	Returns:
		tuple: A tuple of interpolated time values and corresponding y values (ts_interp, ys_interp). *Note that ys_interp will have the same shape as ys*
	"""
	ts_interp = np.linspace(ts[0], ts[-1], n_points)
	ys_interp = None

	if len(ys.shape) == 1:
		# only a single array of y-values was passed
		f = interpolate.PchipInterpolator(ts, ys)
		ys_interp = f(ts_interp)
	else:
		ys_interp = []
		for y in ys:
			f = interpolate.PchipInterpolator(ts, y)
			ys_interp.append( f(ts_interp) )
		ys_interp = np.asarray(ys_interp)

	return ts_interp, ys_interp

def clean_time_series_features(ts:np.ndarray, ys:np.ndarray) -> tuple:
	"""Removes duplicate timestamps and corresponding entries in ys

	Args:
		ts (np.ndarray): array of time values corresponding to every entry in ys
		ys (np.ndarray): a single array (or several stacked arrays) of values corresponding to ts

	Returns:
		tuple: (ts_clean, ys_clean). *Note that ys_clean will have the same data type as ys*
	"""
	ts_clean, idxs = np.unique(ts, return_index=True)
	ys_clean = None
	if len(ys.shape) == 1:
		ys_clean = ys[idxs]
	else:
		ys_clean = []
		for y in ys:
			ys_clean.append( y[idxs] )
		ys_clean = np.asarray(ys_clean)

	return ts_clean, ys_clean

def get_df_test_tracker(dataset_id:str):
	assert dataset_id in get_available_dataset_ids()
	return pd.read_excel(
		dir_spreadsheets.joinpath("Cell Test Tracker.xlsx"),
		sheet_name=dataset_id,
		engine='openpyxl')

def get_df_VvSOC(dataset_id:str):
	assert dataset_id in get_available_dataset_ids()
	return pd.read_excel(dir_spreadsheets.joinpath("V v SOC Data.xlsx"), sheet_name=f"{dic_available_dataset_info[dataset_id]['cell_chemistry']} 1C V vs SOC", engine='openpyxl')


def get_available_dataset_ids() -> list:
	"""
	Returns the key names of all available datasets. 
	You can access more information of the dataset by using 'dic_available_dataset_info[key]' where 'key' is an item in the list returned by this function.
	"""
	return list(dic_available_dataset_info.keys())

def all_cell_ids_in_dataset(dataset_id:str) -> np.ndarray:
	"""Return numpy array of all cell ids in the specified dataset

	Args:
		dataset_id (str): Must be a valid dataset_id. Use 'get_available_dataset_ids()' to get all valid ids.

	Returns:
		np.ndarray: Array of all cell ids contained in the dataset
	"""
	assert dataset_id in get_available_dataset_ids()
	df = get_df_test_tracker(dataset_id)
	assert 'Cell ID' in df.columns.values, f"Could not find a 'Cell ID' column in the test tracker for the \'{dataset_id}\' dataset."
	cells = df['Cell ID'].unique() 
	return cells

def get_group_id(dataset_id:str, cell_id:int) -> int:
	"""Obtains the group id corresponding to the given cell id and dataset id

	Args:
		dataset_id (str): Must be a valid dataset id. Use 'get_available_dataset_ids()' to view available dataset ids
		cell_id (int): cell id to find group for

	Returns:
		int: group id corresponding to the given cell id
	"""
	assert dataset_id in get_available_dataset_ids()
	df = get_df_test_tracker(dataset_id)
	try:
		temp = df.loc[df['Cell ID'] == cell_id, 'Group ID'].values
	except KeyError:
		raise RuntimeError(f"Could not find a 'Group ID' column in the test tracker for the \'{dataset_id}\' dataset.")
	assert len(temp) == 1, "Could not find the group id for this cell"
	return int(temp[0])

def get_cell_ids_in_group(dataset_id:str, group_id:int) -> np.ndarray:
	"""Gets all cell ids in the specified group id and dataset id

	Args:
		dataset_id (str): Must be a valid dataset id. Use 'get_available_dataset_ids()' to view available dataset ids
		group_id (int): The id of the group for which to return the cell ids 

	Returns:
		np.ndarray: An array of all cell_ids in the specified group
	"""

	assert dataset_id in get_available_dataset_ids()
	df = get_df_test_tracker(dataset_id)
	assert 'Group ID' in df.columns.values, f"Could not find a 'Group ID' column in the test tracker for the \'{dataset_id}\' dataset."
	assert 'Cell ID' in df.columns.values, f"Could not find a 'Cell ID' column in the test tracker for the \'{dataset_id}\' dataset."

	cells = df.loc[df['Group ID'] == group_id, 'Cell ID'].unique()
	return cells

def get_group_ids_from_cells(dataset_id:str, cells):
	"""Gets all group ids from the specified list of cell ids

	Args:
		dataset_id (str): Must be a valid dataset id. Use 'get_available_dataset_ids()' to view available dataset ids
		cells (list | np.ndarray): The list of cell ids 

	Returns:
		np.ndarray: An array of all unique group_ids for the specified cells
	"""
	assert dataset_id in get_available_dataset_ids()
	
	all_groups = [get_group_id(dataset_id, c) for c in cells]
	return np.unique(np.asarray(all_groups))




def get_preprocessed_data_files(dataset_id:str, data_type:str, cell_id:int):
	"""Returns a list of Path objects to all pkl files containing data for this cell

	Args:
		dataset_id (str): Must be a valid dataset_id. Use 'get_available_dataset_ids()' to get all valid ids.
		data_type (str): {'rpt', 'cycling'}. Whether to look for RPT or Cycling data
		cell_id (int): The cell id to find data for

	Returns:
		list: list of Path objects
	"""
	assert dataset_id in get_available_dataset_ids()
	assert data_type in ['rpt', 'cycling']

	dir_data = dic_available_dataset_info[dataset_id]['path_downloaded_data'].joinpath(f'{data_type}_data')
	all_files = list(dir_data.glob(f'{data_type}_cell_{int(cell_id):02d}*'))

	def _file_part_num(file_path:Path):
		
		file_str = str(file_path.name)
		return int(file_str[file_str.rindex('_part') + len('_part') : file_str.rindex(file_path.suffix)])
	
	if len(all_files) == 0 or '_part' not in str(all_files[0]): 
		return all_files
	else:
		return sorted(all_files, key=_file_part_num)

def load_preprocessed_data(file_paths) -> pd.DataFrame:
	"""Loads the processed data contained at the provided file path(s). Use 'get_preprocessed_data_files()' to get all file paths

	Args:
		file_paths (Path or list): Path or list of Path objects. 

	Returns:
		pd.DataFrame: A dataframe containing the data at the provided filepath .If multiple file paths are provided, the data will be concatenated into a single dataframe
	"""

	if hasattr(file_paths, '__len__'):
		all_data = []
		if len(file_paths) == 0:
			print("WARNING: The provided list of filepaths is empty. Returning None")
			return None
		for file_path in file_paths:
			if file_path.suffix == '.pkl':
				all_data.append( pickle.load(open(file_path, 'rb')) )
			elif file_path.suffix == '.csv':
				all_data.append( pd.read_csv(file_path, index_col=0) )
			else:
				raise TypeError(f"File suffix ({file_path.suffix}) not supported.")
		return pd.concat(all_data, ignore_index=True)
	else:
		if file_paths.suffix == '.pkl':
			return pickle.load(open(file_paths, 'rb'))
		elif file_paths.suffix == '.csv':
			return pd.read_csv(file_paths, index_col=0)
		else:
			raise TypeError(f"File suffix ({file_paths.suffix}) not supported.")

def get_processed_data(dataset_id:str, data_type:str, filename:str=None) -> dict:
	"""Loads and returns the saved processed data for the specified data type

	Args:
		dataset_id (str): Must be a valid dataset_id. Use 'get_available_dataset_ids()' to get all valid ids.
		data_type (str): {'cc', 'slowpulse', }. The type of processed data to load. 
		filename (str, optional): Can optionally specify the filename of the saved data. If not provided, the most recent auto-named file will be returned.

	Returns:
		dict: The saved data
	"""

	assert dataset_id in get_available_dataset_ids()
	assert data_type in ['cc', 'slowpulse']

	dir_data = dir_processed_data.joinpath(dataset_id)

	if filename is not None:
		f = dir_data.joinpath(filename)
		if not f.exists(): 
			raise ValueError(f"Could not find specified file: {f}")
		return pickle.load(open(f, 'rb'))
	else:
		prev_files = sorted(dir_data.glob(f"data_{data_type}_*.pkl"))
		if len(prev_files) == 0: 
			raise ValueError("Could not find any previously saved files. Try providing a filename")
		else:
			return pickle.load(open(prev_files[-1], 'rb'))



def get_features_from_pulse_data_v1(pulse_data:dict, pulse_type:str, pulse_soc=None) -> tuple:
	"""Return (features, targets, cell_ids) corresponding to the filtered data. Features contains the relative pulse voltage (100 data points). Targets contains the singular q_dchg value.

	Args:
		pulse_data (dict): The postprocessed slowpulse data
		pulse_type (str): Whether to filt to 'chg' or 'dchg' pulses
		pulse_soc (int, optional): If specified will additionally filter data to the single SOC value. Defaults to None.

	Returns:
		tuple: Returns (features, targets, cell_ids)
	"""
	assert pulse_type in ['chg', 'dchg']

	#region: filter data to pulse_type and pulse_soc (use all socs if pulse_soc=None)
	filt_idxs = None
	if pulse_soc is None:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type))
	else:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type) & (pulse_data['soc'] == pulse_soc))
	#endregion

	features = np.asarray([v - v[0] for v in pulse_data['voltage'][filt_idxs]])
	targets = pulse_data['q_dchg'][filt_idxs].reshape(-1,1)
	cell_ids = pulse_data['cell_id'][filt_idxs].reshape(-1,1)
	return features, targets, cell_ids

def get_features_from_pulse_data_v2(pulse_data:dict, pulse_type:str, pulse_soc=None) -> tuple:
	"""Return (features, targets, cell_ids) corresponding to the filtered data. Features contains the relative pulse voltage (100 data points). Targets contains the singular SOH value.

	Args:
		pulse_data (dict): The postprocessed slowpulse data
		pulse_type (str): Whether to filt to 'chg' or 'dchg' pulses
		pulse_soc (int, optional): If specified will additionally filter data to the single SOC value. Defaults to None.

	Returns:
		tuple: Returns (features, targets, cell_ids)
	"""
	assert pulse_type in ['chg', 'dchg']

	#region: filter data to pulse_type and pulse_soc (use all socs if pulse_soc=None)
	filt_idxs = None
	if pulse_soc is None:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type))
	else:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type) & (pulse_data['soc'] == pulse_soc))
	#endregion

	features = np.asarray([v - v[0] for v in pulse_data['voltage'][filt_idxs]])
	targets = pulse_data['soh'][filt_idxs].reshape(-1,1)
	cell_ids = pulse_data['cell_id'][filt_idxs].reshape(-1,1)
	return features, targets, cell_ids

def get_features_from_pulse_data_v3(pulse_data:dict, pulse_type:str, pulse_soc=None) -> tuple:
	"""Return (features, targets, cell_ids) corresponding to the filtered data. Features contains the raw pulse voltage (100 data points). Targets contains the singular SOH value.

	Args:
		pulse_data (dict): The postprocessed slowpulse data
		pulse_type (str): Whether to filt to 'chg' or 'dchg' pulses
		pulse_soc (int, optional): If specified will additionally filter data to the single SOC value. Defaults to None.

	Returns:
		tuple: Returns (features, targets, cell_ids)
	"""
	assert pulse_type in ['chg', 'dchg']

	#region: filter data to pulse_type and pulse_soc (use all socs if pulse_soc=None)
	filt_idxs = None
	if pulse_soc is None:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type))
	else:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type) & (pulse_data['soc'] == pulse_soc))
	#endregion

	features = pulse_data['voltage'][filt_idxs]
	targets = pulse_data['soh'][filt_idxs].reshape(-1,1)
	cell_ids = pulse_data['cell_id'][filt_idxs].reshape(-1,1)
	return features, targets, cell_ids

def get_features_from_pulse_data_v4(pulse_data:dict, pulse_type:str, pulse_soc=None) -> tuple:
	"""Return (features, targets, cell_ids) corresponding to the filtered data. Features contains the following manually defined features: {'mean', 'median', 'variance', 'area', 'endpoints', first 5 PCA}. Targets contains the singular SOH value.

	Args:
		pulse_data (dict): The postprocessed slowpulse data
		pulse_type (str): Whether to filt to 'chg' or 'dchg' pulses
		pulse_soc (int, optional): If specified will additionally filter data to the single SOC value. Defaults to None.

	Returns:
		tuple: Returns (features, targets, cell_ids)
	"""
	assert pulse_type in ['chg', 'dchg']

	#region: filter data to pulse_type and pulse_soc (use all socs if pulse_soc=None)
	filt_idxs = None
	if pulse_soc is None:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type))
	else:
		filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type) & (pulse_data['soc'] == pulse_soc))
	#endregion

	voltages = np.asarray([v - v[0] for v in pulse_data['voltage'][filt_idxs]])
	mean = np.mean(voltages, axis=1).reshape(-1,1)
	median = np.median(voltages, axis=1).reshape(-1,1)
	variance = np.var(voltages, axis=1).reshape(-1,1)
	area = np.sum(voltages, axis=1).reshape(-1,1)
	endpoints = voltages[:,[0, 29,30, 39,40, 99]]
	pca = PCA(n_components=5)
	pca.fit(voltages)
	pca_vals = pca.transform(voltages)

	features = np.concatenate([mean, median, variance, area, endpoints, pca_vals], axis=1)
	targets = pulse_data['soh'][filt_idxs].reshape(-1,1)
	cell_ids = pulse_data['cell_id'][filt_idxs].reshape(-1,1)
	return features, targets, cell_ids
