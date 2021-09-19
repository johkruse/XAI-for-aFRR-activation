# Explainable Machine Learning for secondary control activation in Germany
Code accompanying the mansucript "Secondary control activation analysed and predicted with explainable AI".
Preprint: <http://arxiv.org/abs/2109.04802>


## Install

The code is written in Python (tested with python 3.7). To install the required dependencies execute the following commands:

```[python]
python3.7 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Usage

The `scripts` folder contains scripts to create the paper results and `notebooks` contains a notebook to reproduce the paper figures. The `utils` folder comprises one module to process ENTSO-E data.

The `scripts` contain a pipeline of six different stages:

* `1_download_data.sh`: A bash script to download the external features from the ENTSO-E Transparency Platform. 
* `2_afrr_data_prep.py`: Process the data files for activated aFRR and tendered aFRR demands.
* `3_entsoe_data_prep.py`: Collect and aggregate external features for Germany and other IGCC member states.
* `4_external_feature_prep.py`: Add additional engineered features to the set of external features and define model types with different inputs.
* `5_train_test_split.py`: Split data set into train and test set and save data in a version folder.
* `6_model_fit.py`: Fit the LightGBM model, optimize hyper-parameters and calculate SHAP values.


## Input data and results
All the raw data is publicly available and we have uploaded the processed data and our results on zenodo. The data and the (intermediate) results can be used to run the scripts.

* **External features, aFRR data, and results of hyper-parameter optimization and model interpretation**: The output of scripts 2 to 6 are available on the accompanying [zenodo repository](https://doi.org/10.5281/zenodo.5497500). The data is assumed to reside in the repository directory within `./data/` and the results should reside in `./results/`. In particular, the data of external features and aFRR activation can be used to re-run the model fit. 
* **Raw aFRR data**: We have used publicly available [aFRR data](regelleistung.net/) as an input to `2_stability_indicator_prep.py`. The raw aFRR data files are also available on our accompanying [zenodo repository](https://doi.org/10.5281/zenodo.5497500). The raw aFRR data is supposed to reside in `./data/DE/frr_data/`. 
* **Raw ENTSO-E data**: The output of `1_download_data.sh` is not available on the zenodo repository, but can be downloaded from the [ENTSO-E Transparency Platform](transparency.entsoe.eu/) via the bash script. The ENTSO-E data is assumed to reside in `../../External_data/ENTSO-E` relative to this code repository.

