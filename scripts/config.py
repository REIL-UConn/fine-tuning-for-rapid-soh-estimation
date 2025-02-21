
import sys, os, warnings, re, pickle
from pathlib import Path
import pandas as pd
import numpy as np

# PLEASE FILL IN THE TWO DICTIONARY VALUES MARKED WITH 'TODO' BELOW

# =================================================================================
#   AVAILABLE DATASETS
# =================================================================================
dic_available_dataset_info = {
    'UConn-ILCC-LFP': {
        'dataset_fullname':'UConn-ISU-ILCC LFP battery aging',
        'cell_chemistry':'LFP',
        'cell_count':64,
        'group_count':11,
        'pulse_socs_tested':[20,50,90],
        'path_downloaded_data': None,   # TODO: set this path to the downloaded data
        # Ex: Path('.../UConn-ISU-ILCC LFP Aging Dataset')
    },
    'UConn-ILCC-NMC': {
        'dataset_fullname':'UConn-ILCC NMC battery aging',
        'cell_chemistry':'NMC',
        'cell_count':44,
        'group_count':11,
        'pulse_socs_tested':np.arange(10,91,10),
        'path_downloaded_data': None,   # TODO: set this path to the downloaded data
        # Ex: Path('.../UConn-ILCC NMC Aging Dataset')
        },
}


# =================================================================================
#   PATH DEFINITONS
# =================================================================================
cwd = os.path.abspath(__file__)
dir_repo_main = Path(str(cwd)[:str(cwd).rindex('fine-tuning-for-rapid-soh-estimation') + len('fine-tuning-for-rapid-soh-estimation')])
assert dir_repo_main.is_dir()
dir_figures = dir_repo_main.joinpath("figures")
dir_notebooks = dir_repo_main.joinpath("notebooks")
dir_processed_data = dir_repo_main.joinpath("processed_data")
dir_results = dir_repo_main.joinpath("results")
dir_spreadsheets = dir_repo_main.joinpath("spreadsheets")


# =================================================================================
#   COLOR PALETTE
# =================================================================================
global_colors = {
    'defaults':{
        'builtin':['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    },
    'by_group':{
        'primary':['#4E79A7', '#F28E2B', '#59A14F', '#D37295', '#499894', 
                   '#E15759', '#363433', '#B6992D', '#B07AA1', '#9D7660', '#637939'],
        'secondary':['#A0CBE8', '#FFBE7D', '#8CD17D', '#FABFD2', '#86BCB6', 
                     '#FF9D9A', '#94908e', '#F1CE63', '#D4A6C8', '#D7B5A6', '#8CA252'],
        'primary_v2':['#5F4690FF', '#1D6996FF', '#38A6A5FF',	'#0F8554FF', '#73AF48FF',
                     '#EDAD08FF', '#E17C05FF', '#CF292BFF', '#D93D7FFF', '#AF1060FF', '#303043FF'],
        'secondary_v2':['#5F469099', '#1D699699', '#38A6A599',	'#0F855499', '#73AF4899',
                     '#EDAD0899', '#E17C0599', '#CF292B99', '#D93D7F99', '#AF106099', '#30304399']
    }
}
