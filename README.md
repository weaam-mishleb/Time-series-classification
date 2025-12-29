# Time-Series Classification Project

## Overview
This is encrypted traffic classification research that represents network traffic as time-series data and uses state-of-the-art (SOTA) transformer models from TSlib (https://github.com/thuml/Time-Series-Library).

## Datasets
- **UTMobile** - https://github.com/YuqiangHeng/UTMobileNetTraffic2021
- **CESNET** - https://zenodo.org/records/7409924
  - GitHub: https://github.com/CESNET/cesnet-datazoo

## Requirements
- Python 3.11 (recommended for the moment model library)

## Code Structure
1. **TimeseriesCreate.py** - Creates the appropriate time-series representation from raw datasets (preferably as CSV files)
2. **TimeSeries_functions.py** - Handles transformer model building, training, evaluation, and preprocessed data fetching
3. **main_program** - Runs TimeSeries_functions.py on all small windows and transformer models

## Additional Resources
- **Notebooks (.ipynb)** - Jupyter notebooks for more specific data exploration and preprocessing tasks (e.g., choosing the optimal big window size)

## Authors
- Amit Dvir
- Chen Hajaj
- Shachar Ketz
- Chanan Helman
