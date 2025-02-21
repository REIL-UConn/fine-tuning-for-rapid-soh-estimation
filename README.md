# fine-tuning-for-rapid-soh-estimation
This repository contains the Python scripts and processed data required to recreate the results and figures presented in the paper: “Fine-tuning for rapid capacity estimation of lithium-ion batteries” ($\textcolor{red}{\text{DOI will be provided soon}}$)

*Questions pertaining to the scripts and data provided in this repository can be directed to Ben Nowacki (benjamin.nowacki@uconn.edu)*


## Paper Details & Abstract

### Fine-tuning for rapid capacity estimation of lithium-ion batteries

Benjamin Nowacki $^{1}$, Thomas Schmitt $^{2}$, Phillip Aquino $^{3}$, Chao Hu $^{1,\dagger}$

&nbsp;&nbsp;&nbsp;&nbsp; $^{1}$ School of Mechanical, Aerospace, and Manufacturing Engineering, University of Connecticut, Storrs, CT 06269

&nbsp;&nbsp;&nbsp;&nbsp; $^{2}$ Honda Research Institute Europe GmbH, D-63073 Offenbach, Germany

&nbsp;&nbsp;&nbsp;&nbsp; $^{3}$ Honda Research Institute US, Columbus, OH, 43212, USA

&nbsp;&nbsp;&nbsp;&nbsp; $^{\dagger}$ Indicates corresponding author. Email: chao.hu@uconn.edu


**Abstract**:

The widespread adoption of large-scale battery-powered technologies, such as electric vehicles and renewable energy storage systems, has led to growing interest in assessing their remaining usability after years of operation.
As these systems age, state-of-health estimation becomes crucial for ensuring reliability and safety, and for extending life through second-use applications.
However, current methods -- spanning physics-based, empirical, and data-driven approaches -- face challenges, including insufficient labeled data, high resource costs, and poor generalizability across diverse usage conditions. 
Data-driven models, in particular, struggle to extrapolate beyond their training domain, limiting their applicability in real-world scenarios.
This work develops a fine-tuning framework to address these challenges, enabling rapid capacity estimation using short-duration ($\leq100$ seconds) features. 
Tested on two battery chemistries (LFP/Gr and NMC/Gr), the fine-tuned model achieves average mean-absolute-percent-errors of $2.592\%$ and $3.094\%$ on datasets collected from the respective chemistries. 
Compared to two baseline approaches, direct-transfer and target-only modeling, fine-tuning achieves a $25\%$ reduction in estimation error in the target domain, on average.
Domain differences are quantified using statistical measures such as Kullback-Leibler divergence and maximum mean discrepancy, which are shown to correlate with fine-tuning performance, offering insights into domain compatibility.
This study also analyzes the impact of feature selection, hyper-parameter tuning, and labeled data availability on fine-tuning efficacy, providing practical guidelines for real-world applications.



## Repository Structure

```
|- fine-tuning-for-rapid-soh-estimation/
    |- notebooks/ 
    |- processed_data/
        |- UConn-ILCC-LFP/
        |- UConn-ILCC-NMC/
    |- results/
    |- scripts/
        |- config.py
        |- data_processing/
            |- common_methods.py
            |- postprocessing.py
        |- modeling/
            |- common_methods.py
            |- fine_tuning.py
    |- spreadsheets/
    |- environment.yml
    |- LICENSE
    |- README.md
```


### `notebooks/`

The notebooks directory contains several Jupyter notebooks that were used to conduct the majority of the analysis. Common methods/functions are refactored to Python scripts contained within the `scripts/` directory. 

* `../data_overview.ipynb`: Provides details on the datasets used, including plots on the aging trajectories, pulse profiles, internal resistance changes, and voltage distributions at discrete time steps within a pulse.
* `../feature_importance.ipynb`: Provides implementation details on conducting feature importance of the pulse on SOH estimation in the source domain and for fine-tuning to all target domains. 
* `../fine_tuning.ipynb`: Provides implementation details on how fine-tuning is conducted between single and multiple source domains to a single target domain. Low-level SOH estimation plots and heatmaps for average target error for all transfer setttings are shown.
* `../information_metrics.ipynb`: Provides details on how collections of pulses conducted at a different SOCs are quantified for their domain differences. Plots are shown for the quanitfied differences across all transfer settings and the correlations between these differences and fine-tuning performance metrics. 
* `../model_optimization.ipynb`: Provides the implementation details on how models were optimized for each cell chemsitry and pulse type using the Optuna framework. Optimization is performed for the model's source test error and fine-tuned models target test error.
* `../phase_transitions.ipynb`: Details how dQ/dV data is extracted from the datasets and used to isolate three distinct phase transitions (for the NMC cells). These phase transitions are then evaluated for domain diffferences and correlated with fine-tuning performance metrics.
  

### `processed_data/`

Note that due to the size of all raw data, only post-processed data is retained in this Git repository. The raw data collected on the 64 LFP/Gr and 44 NMC/Gr cells can be downloaded at: 
* [UConn-ISU-ILCC LFP aging dataset](https://digitalcommons.lib.uconn.edu/reil_datasets/1/)
* [UConn-ILCC NMC aging dataset](https://digitalcommons.lib.uconn.edu/reil_datasets/2/)

Please see the 'README.txt' file contained in the above linked datasets for an overview of how the raw data files are structured.


### `scripts/`

This folder contains scripts pertaining to unique aspects of data processing and analysis.

* `../config.py`
  - Contains a set of common path definitions and a dictionary outlining key information related to each dataset
  - **IMPORTANT**: If you want to use the `postprocessing.py` script, you must download the raw data and set the local paths under the respective keys in `dic_available_dataset_info`. See the two TODO statements
* `../data_processing/common_methods.py`
  - Contains a set of functions commonly used to load the downloaded data, perform postprocessing/feature extraction, and extract cell information.
* `../data_processing/postprocessing.py`
  - Contains all code used to create modeling data (ei, extracting pulse profiles from the RPT data).
  - Running this script will regenerate the pulse data for both datasets (stored under `../processed_data`). Note that this *won't* overwrite previously processed data; it will simply save a new copy. 

* `../modeling/common_methods.py`
  - Contains a set of reused methods across the modeling scripts, including a ModelOptimizer class and optimial hyperparameter lookup from saved results.
* `../modeling/fine_tuning.py`
  - This is the most relevant script pertaining to the paper. It provides all methods related to the fine-tuning process, parameter optimization, and evaluation of many different test conditions.
  - Towards the bottom of the script, you will find the `test_definitions` dictionary. Each unique key defines a sub-dictionary with `'n_iterations'` and `'test_values'` fields. These defined the parameter to be varied and the number of repeat trials per unique value. Each key in `test_definitions` requires a corresponding block of code in the `TransferLearning_Parameter_Generator` generator function. This function produces the corresponding set of data `Splitting_Parameters`, model `FineTuning_Parameters`, and additional `TransferLearning_Parameters`.
  - Running this script will perform fine-tuning for any uncommented defined key in `test_definitions`. Results are saved under the path defined by the `results_folder` variable. See the `if __name__ == '__main__':` block for more details.


### `results/`

This directory contains saved results from several analsyses performed. Files are mostly saved in the binary Pickle format. See `notebooks` and `scripts/` for details on how to properly load and use these results.


### `spreadsheets/`

This directory contains two spreadsheets: 'Cell Test Tracker.xlsx' and 'V vs SOC Data.csv'. The former provides a simple mapping between cell ID, tester details, and cycling conditions for both datasets. The latter is an interpolated lookup table of cell terminal voltage and state-of-charge (defined with coulomb counting) on LFP/Gr and NMC/Gr cells during a 1C and C/30 CC-CV charges.

