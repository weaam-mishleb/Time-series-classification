import os
import pandas as pd
import numpy as np
from collections import Counter
import argparse

def process_TextQuic_traffic(input_file, output_file, window_size):
    """
    Process QUIC traffic data and create an aggregated CSV file with custom window size statistics.
    Includes rows for empty windows where no packets were transferred.
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path where the output CSV will be saved
    window_size (float): Size of the time window in seconds (e.g., 0.01 for 10 ms)
    
    Returns:
    pd.DataFrame: The processed dataframe that was saved to CSV
    """
    # Read the original CSV file
    df = pd.read_csv(input_file)
    print("Input data sample:")
    print(df.head())  # Debug: Print the first few rows of the input data

    # Calculate the window indicator
    df['window'] = (df['Relative time'] // window_size).astype(int)
    print("\nWindow calculation sample:")
    print(df[['Relative time', 'window']].head())  # Debug: Print Relative times and corresponding windows

    # Calculate time from start
    start_time = df['Relative time'].min()
    df['time_from_start'] = df['Relative time'] - start_time

    # Determine the range of windows
    min_window = df['window'].min()
    max_window = df['window'].max()
    print(f"\nWindow range: {min_window} to {max_window}")

    # Create a DataFrame to hold all possible windows
    all_windows = pd.DataFrame({
        'window': range(min_window, max_window + 1)
    })

    # Group by window and calculate statistics
    grouped = df.groupby('window').agg({
        'Size': ['sum', 'count', 'mean'],
        'Direction': lambda x: Counter(x).most_common(1)[0][0] if not x.empty else np.nan
    }).reset_index()

    # Flatten the multi-level columns from the aggregation
    grouped.columns = ['window', 'total_size', 'packets_count', 'average_size', 'most_common_direction']

    # Merge with all_windows to include empty windows
    result_df = pd.merge(all_windows, grouped, on='window', how='left')

    # Fill NaN values for empty windows
    result_df['total_size'] = result_df['total_size'].fillna(0)
    result_df['packets_count'] = result_df['packets_count'].fillna(0)
    result_df['average_size'] = result_df['average_size'].fillna(0)
    result_df['most_common_direction'] = result_df['most_common_direction'].fillna(-1)  # Use -1 for no direction

    # Calculate additional statistics
    result_df['time_from_start'] = result_df['window'] * window_size
    result_df['exact_time'] = (window_size * result_df['window']) + df['Timestamp'].min()
    result_df['direction_0_ratio'] = result_df.apply(
        lambda row: (row['packets_count'] - row['most_common_direction']) / row['packets_count'] if row['packets_count'] > 0 else 0,
        axis=1
    )
    result_df['direction_1_ratio'] = result_df.apply(
        lambda row: row['most_common_direction'] / row['packets_count'] if row['packets_count'] > 0 else 0,
        axis=1
    )
    # result_df['packets_per_second'] = result_df['packets_count'] / window_size

    
    # Sort by window
    result_df = result_df.sort_values('window').reset_index(drop=True)

    # Limit to 30000 rows and save to CSV
    if len(result_df) > 30000:        
        print(f"\nWarning: Output truncated to 30000 rows from {len(result_df)} rows")
        result_df = result_df.head(30000)
    result_df.to_csv(output_file, index=False)
    

    # Print some summary statistics
    total_intervals = len(result_df)
    print(f"\nProcessed {len(df)} packets into {total_intervals} {window_size}-second intervals")
    print(f"\nTime range: {result_df['exact_time'].min()} to {result_df['exact_time'].max()}")
    print(f"Total duration: {(result_df['exact_time'].max() - result_df['exact_time'].min() + window_size):.3f} seconds")
    print(f"\nAverage packets per window: {result_df['packets_count'].mean():.2f}")
    print(f"Max packets in one window: {result_df['packets_count'].max()}")
    print(f"Min packets in one window: {result_df['packets_count'].min()}")

    # Print example of how the data is split
    print("\nExample of window splitting:")
    print(result_df[['window', 'packets_count', 'total_size']].head(6))

    return result_df

def pipeline(input_dir, output_dir, window_size):
    """
    Process all QUIC traffic data files in a directory and save aggregated CSV files.
    
    Parameters:
    input_dir (str): Path to the directory containing input CSV files
    output_dir (str): Path to the directory where output CSV files will be saved
    window_size (float): Size of the time window in seconds (e.g., 0.01 for 10 ms)
    """
     # Ensure the directory exists
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    # List all CSV files in the input directory
    files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Process each file
    for file in files:
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, f"timeseries_{file}")
        print(f"\nProcessing file: {input_file}")
        process_quic_traffic(input_file, output_file, window_size)
    return

def create_direction_dataset(input_dirs, output_dir, window_size=4):
    """
    Create a dataset from the first window_size timesteps of each sample, organized by direction.
    Saves all samples in two files: features.csv (X) and labels.csv (y)
    
    Parameters:
    input_dirs (list): List of Directories containing processed timeseries CSV files
    output_dir (str): Directory to save the processed samples
    window_size (int): Number of timesteps to keep from each sample
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return
    
    # Initialize lists to store all samples
    all_features = []
    all_labels = []
    
    for i, dir in enumerate(input_dirs):
        files = [file for file in os.listdir(dir) if file.startswith('timeseries_') and file.endswith('.csv')]
        
        for file in files:
            input_path = os.path.join(dir, file)
            df = pd.read_csv(input_path)
            
            # Skip if time range is too small
            max_time = df['time_from_start'].max()
            if max_time < window_size:
                print(f"Skipping {file}: time range too short ({max_time} < {window_size})")
                continue
            
            # Take samples up to window_size time
            sample = df[df['time_from_start'] <= window_size]
            flatten_sample = sample.values.flatten()
            # Store features (all columns)
            all_features.append(flatten_sample)
            # Store label (directory index)
            all_labels.append(pd.Series(i))
    
    # Convert lists to DataFrame
    X = pd.DataFrame(all_features)
    y = pd.Series(all_labels)
    
    # Save combined datasets
    X.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    y.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
    
    print(f"Saved {len(all_features)} samples to {output_dir}")
            

# Example usage
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process QUIC traffic data and create an aggregated CSV file with custom window size statistics.")
    # parser.add_argument("--input", required=True, help="Path to the input CSV file")
    # parser.add_argument("--output", required=True, help="Path where the output CSV will be saved")
    # parser.add_argument("--window_size", type=float, required=True, help="Size of the time window in seconds (e.g., 0.01 for 10 ms)")
    # args = parser.parse_args()
    # input_files = args.input
    # output_file = args.output
    # window_size = args.window_size

    # Process the data
    # aggregated_df = process_quic_traffic(input_file, output_file, window_size)
    # # pipeline(input_file, output_file, window_size)

    # # # Display the first few rows of the processed data
    # # print("\nFirst few rows of the processed data:")
    # # print(aggregated_df.head())

    # input_files0 = ['pretraining/Google Doc/250ms', 'pretraining/Google Drive/250ms','pretraining/Google Music/250ms','pretraining/Google Search/250ms','pretraining/Youtube/250ms']
    # input_files1 = ['pretraining/Google Doc/5ms', 'pretraining/Google Drive/5ms','pretraining/Google Search/5ms','pretraining/Youtube/5ms']
    # input_files2 =  ['pretraining/Google Doc/5ms', 'pretraining/Google Drive/5ms','pretraining/Youtube/5ms']
    # output_file1 = r'C:\Users\shach\Desktop\project FlowPic\big_project\QUIC Dataset\pretraining\Dataset\5ms\10\4 classes (without g.m)'
    # output_file2 = r'C:\Users\shach\Desktop\project FlowPic\big_project\QUIC Dataset\pretraining\Dataset\5ms\10\3 classes (without g.m & g.s)'
    # window_size = 10

    # create_direction_dataset(input_files1, output_file1, window_size)
    # create_direction_dataset(input_files2, output_file2, window_size)
    # Create QuicText DataSet:
    intput_dirs_10 = ["../../../data/Chanan/QuicText/timeseries/10/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/10/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/10/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/10/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/10/Youtube"
                     ]
    intput_dirs_20 = ["../../../data/Chanan/QuicText/timeseries/20/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/20/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/20/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/20/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/20/Youtube"
                      ]
    intput_dirs_30 = ["../../../data/Chanan/QuicText/timeseries/30/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/30/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/30/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/30/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/30/Youtube"
                      ]
    intput_dirs_40 = ["../../../data/Chanan/QuicText/timeseries/40/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/40/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/40/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/40/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/40/Youtube"
                      ]
    intput_dirs_75 = ["../../../data/Chanan/QuicText/timeseries/75/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/75/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/75/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/75/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/75/Youtube"
                      ]
    intput_dirs_100 = ["../../../data/Chanan/QuicText/timeseries/100/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/100/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/100/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/100/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/100/Youtube"
                      ]
    intput_dirs_150 = ["../../../data/Chanan/QuicText/timeseries/150/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/150/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/150/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/150/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/150/Youtube"
                      ]
    intput_dirs_200 = ["../../../data/Chanan/QuicText/timeseries/200/Google Doc",
                      "../../../data/Chanan/QuicText/timeseries/200/Google Drive",
                      "../../../data/Chanan/QuicText/timeseries/200/Google Music",
                      "../../../data/Chanan/QuicText/timeseries/200/Google Search",
                      "../../../data/Chanan/QuicText/timeseries/200/Youtube"
                      ]
    
    # input_dir = "../../..//data/Chanan/TextQuic/csv"
    output_dir_10 = "../../../data/Chanan/QuicText/datasets/10ms/5s/5 classes"
    output_dir_20 = "../../../data/Chanan/QuicText/datasets/20ms/5s/5 classes"
    output_dir_30 = "../../../data/Chanan/QuicText/datasets/30ms/5s/5 classes"
    output_dir_40 = "../../../data/Chanan/QuicText/datasets/40ms/5s/5 classes"
    output_dir_75 = "../../../data/Chanan/QuicText/datasets/75ms/5s/5 classes"
    output_dir_100 = "../../../data/Chanan/QuicText/datasets/100ms/5s/5 classes"
    output_dir_150 = "../../../data/Chanan/QuicText/datasets/150ms/5s/5 classes"
    output_dir_200 = "../../../data/Chanan/QuicText/datasets/200ms/5s/5 classes"

    # intput_dirs = [intput_dirs_20, intput_dirs_30, intput_dirs_40, intput_dirs_75, intput_dirs_100, intput_dirs_150, intput_dirs_200]
    # output_dirs = [output_dir_20, output_dir_30, output_dir_40, output_dir_75, output_dir_100, output_dir_150, output_dir_200]
    

    create_direction_dataset(intput_dirs_10, output_dir_10, window_size=5)
    create_direction_dataset(intput_dirs_20, output_dir_20, window_size=5)
    create_direction_dataset(intput_dirs_30, output_dir_30, window_size=5)
    create_direction_dataset(intput_dirs_40, output_dir_40, window_size=5)
    create_direction_dataset(intput_dirs_75, output_dir_75, window_size=5)
    create_direction_dataset(intput_dirs_100, output_dir_100, window_size=5)
    create_direction_dataset(intput_dirs_150, output_dir_150, window_size=5)
    create_direction_dataset(intput_dirs_200, output_dir_200, window_size=5)