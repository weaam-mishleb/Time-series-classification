# %%
if __name__ == '__main__':
    IS_MOMENT_ENVIRONMENT = False
    # IS_MOMENT_ENVIRONMENT = True

    if IS_MOMENT_ENVIRONMENT:
        import moment_function as mf
        import warnings
        warnings.filterwarnings("ignore", message=".*use_reentrant.*")
        warnings.filterwarnings("ignore", message=".*requires_grad=True.*")
    else:
        import TimeSeries_functions as ts
        #Import models
        from models.TimesNet import Model as TimesNet
        from models.Nonstationary_Transformer import Model as NST
        from models.Informer import Model as Informer
        from models.Autoformer import Model as Autoformer
        from models.FEDformer import Model as FEDformer
        from models.TimeMixer import Model as Timemixer
    
    import torch

    

    from tqdm import tqdm, trange

    from torch.utils.data import DataLoader,TensorDataset
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # compute weight-class
    from sklearn.utils.class_weight import compute_class_weight

    import copy
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import gc
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime, timedelta
    import time
    import traceback
    import subprocess
    import re

    # %%
    if IS_MOMENT_ENVIRONMENT:
        mf.print_gpu_info()
        best_gpu = mf.get_least_used_gpu()
        print(f"Best GPU: {best_gpu}")
    else:
        ts.print_gpu_info()
        best_gpu = ts.get_least_used_gpu()
        print(f"Best GPU: {best_gpu}")

    # %%
    # dataset_name="Cesnet"
    # small_windows = ["10ms"]

    # datasets_dict = ts.prepare_datasets(dataset_name=dataset_name,
    #                                     small_windows=small_windows)
    # (X,y) = datasets_dict['dir_Cesnet_10ms']
    # print(X.shape, y.shape)
    # checkpoint_path = '/home/chanan/Time-Series-Library/checkpoints/Cesnet/18/informer_classifier_500_best_train.pth'
    # model, optimizer, start_epoch, best_metrics, config = ts.load_checkpoint(checkpoint_path)

    # train_loader, val_loader, class_weights = ts.prepare_data(X, y, config)

    # #initialize listener
    # if IS_MOMENT_ENVIRONMENT:
    #     # Initialize user control dictionary
    #     mf.start_keyboard_listener()
    #     # Reset user controls before training this dataset
    #     mf.user_control['stop'] = False
    #     mf.user_control['reduce_lr'] = False
    #     mf.user_control['done'] = False
    #     mf.user_control['wait_and_reduce'] = False
    # else:
    #     # Initialize user control dictionary
    #     ts.start_keyboard_listener()
    #     # Reset user controls before training this dataset
    #     ts.user_control['stop'] = False
    #     ts.user_control['reduce_lr'] = False
    #     ts.user_control['done'] = False
    #     ts.user_control['wait_and_reduce'] = False
    # start_time = time.time()
    # # Continue training
    # model, val_acc, eval_time_ms, updated_metrics = ts.train_classifier(
    #     train_loader,
    #     val_loader,
    #     config,
    #     model=model,
    #     optimizer=optimizer,
    #     start_epoch=start_epoch,
    #     best_train_acc=best_metrics['train_acc'],
    #     class_weights=class_weights
    # )
    
    # training_time = time.time() - start_time

    # # Save evaluation time to CSV
    # results_df = pd.DataFrame({
    #     'model': ["informer"],
    #     'dataset': [dataset_name],
    #     'window_size': ["10ms"],
    #     'eval_time_ms': [eval_time_ms],
    #     'training_time': str(timedelta(seconds=int(training_time))),
    #     'train_accuracy': updated_metrics['train_accuracy'],
    #     'train_precision': updated_metrics['train_precision'],
    #     'train_recall': updated_metrics['train_recall'],
    #     'train_f1': updated_metrics['train_f1'],
    #     'val_accuracy': updated_metrics['val_accuracy'],
    #     'val_precision': updated_metrics['val_precision'],
    #     'val_recall': updated_metrics['val_recall'],
    #     'val_f1': updated_metrics['val_f1'],
    # })
    
    # # Ensure the directory exists
    # csv_dir = os.path.dirname('results/Cesnet/')
    # if not os.path.exists(csv_dir):
    #     os.makedirs(csv_dir)

    # csv_path = 'results/Cesnet/informer_dir_cesnet_10ms_eval_times.csv'
    # if os.path.exists(csv_path):
    #     results_df.to_csv(csv_path, mode='a', header=False, index=False)
    # else:
    #     results_df.to_csv(csv_path, index=False)
    

    # %%
    # dataset_name="Cesnet"
    # small_windows = ["5ms","10ms","20ms","30ms","40ms","50ms","75ms","100ms","150ms","200ms","250ms"]
    # dataset_type = "balanced"
    # datasets_dict = ts.prepare_datasets(dataset_name=dataset_name,
    #                                     small_windows=small_windows,
    #                                     dataset_type=dataset_type)
    # feature_name='vec[upstream,downstream,ratio]'
    # results_1 = ts.evaluate_models_on_datasets(datasets_dict=datasets_dict,
    #                                             model_types=['timesnet'],
    #                                             use_class_weights=False,
    #                                             dataset_name=dataset_name,
    #                                             feature_name=feature_name
    #                                             )
    
    # %%
    dataset_name="Cesnet"
    dataset_type = "dir" #"balanced"
    small_window5 = ["10ms"] #"5ms",
    small_window4 = ["20ms"] #"30ms"
    small_window3 = ["40ms"]#"50ms"
    small_window2 = ["75ms"]#,"100ms"
    small_window1 = ["150ms","200ms"]#,"250ms"
    small_windows =[small_window5] # small_window1,small_window2,small_window3,small_window4,
    feature_name='vec[upstream,downstream,ratio]'
    # feature_name='PacketAmount'
    for i in range(len(small_windows)):
        small_window = small_windows[i]
        print(f"Small window: {small_window}")
        datasets = ts.prepare_datasets(dataset_name=dataset_name,
                                        small_windows=small_window,
                                        dataset_type=dataset_type)
        moment_model = ['moment']
        model_types_1 = ['timesnet'] #,'nst', 'informer'
        model_types_2 = ['autoformer']# 'timesnet',
        model_types_3 = ['fedformer','timemixer']
        results_1 = ts.evaluate_models_on_datasets(datasets_dict=datasets,
                                                model_types= model_types_1, #moment_model model_types_1
                                                use_class_weights=False,
                                                dataset_name=dataset_name,
                                                feature_name=feature_name)
        # results_2 = ts.evaluate_models_on_datasets(datasets_dict=datasets,
        #                                         model_types=model_types_2,
        #                                         use_class_weights=False,
        #                                         dataset_name=dataset_name,
        #                                         feature_name=feature_name
        #                                         )
        # results_3 = ts.evaluate_models_on_datasets(datasets_dict=datasets,
        #                                         model_types=model_types_3,
        #                                         use_class_weights=False,
        #                                         dataset_name=dataset_name,
        #                                         feature_name=feature_name)