# Time-Series Classification 
## This is still a Draft

Encrypted traffic calssification research by representing the netwirk traffic as time-series, and using SOTA transformers models, from TSlib https://github.com/thuml/Time-Series-Library

Datasets: 
1. UTMobile - https://github.com/YuqiangHeng/UTMobileNetTraffic2021 
2. CESNET - https://zenodo.org/records/7409924, github: https://github.com/CESNET/cesnet-datazoo


requierment: (for moment model library use python 3.11)

Code:
1.TimeseriesCreate.py - creating the suit time-series representation from the raw dataset (prefered raw as cvs files)
2.TimeSeries_funtions.py - transformer model building train and evaluation and fetching the preproccesed data  
3. main_program - running the the TimeSeries_funtions.py on all small_windows and transformers model

Addition:
Notebooks (ipynb) for more specific data exploration and preproccessing (e.g., chosing the big window size)

Paper:

Authors: Amit Dvir, Chen Hajaj, Shachar Ketz and CHanan Helman
