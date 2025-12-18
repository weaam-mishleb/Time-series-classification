import os
import pandas as pd
import numpy as np
from collections import Counter
import random
import argparse
import glob
import json
import time

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

def process_VisQuic_traffic(input_file, output_file, window_size):
    """
    Process QUIC traffic data and create an aggregated CSV file with custom window size statistics.
    Skips files that contain NaN values in critical columns.

    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path where the output CSV will be saved
    window_size (float): Size of the time window in seconds (e.g., 0.01 for 10 ms)

    Returns:
    pd.DataFrame or None: The processed dataframe if successful, otherwise None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"\nProcessing file: {input_file}")

        # Ensure required columns exist
        required_columns = ["frame.time_relative", "frame.len", "ip.src", "ip.dst"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Skipping {input_file}: Missing columns {missing_columns}")
            return None

        # Drop rows with NaN in critical columns
        df = df.dropna(subset=["frame.time_relative", "frame.len"])

        if df.empty:
            print(f"Skipping {input_file}: No valid rows after NaN removal.")
            return None

        # Compute window index
        df['window'] = (df['frame.time_relative'] // window_size).astype(int)

        # Calculate time from start
        start_time = df['frame.time_relative'].min()
        df['time_from_start'] = df['frame.time_relative'] - start_time

        # Determine the range of windows
        if df['window'].isna().any():
            print(f"Skipping {input_file}: 'window' calculation resulted in NaN values.")
            return None

        min_window = df['window'].min()
        max_window = df['window'].max()

        if pd.isna(min_window) or pd.isna(max_window):
            print(f"Skipping {input_file}: Window range calculation resulted in NaN values.")
            return None

        print(f"Window range: {min_window} to {max_window}")

        # Create a DataFrame to hold all possible windows
        all_windows = pd.DataFrame({'window': range(int(min_window), int(max_window) + 1)})

        # Assign a direction based on IP
        df['Direction'] = df['ip.src'].apply(lambda x: 0 if x == "0.0.0.0" else 1)

        # Group by window and calculate statistics
        grouped = df.groupby('window').agg({
            'frame.len': ['sum', 'count', 'mean'],
            'Direction': lambda x: Counter(x).most_common(1)[0][0] if not x.empty else np.nan
        }).reset_index()

        # Flatten multi-level columns
        grouped.columns = ['window', 'total_size', 'packets_count', 'average_size', 'most_common_direction']

        # Merge with all_windows to include empty windows
        result_df = pd.merge(all_windows, grouped, on='window', how='left').fillna(0)

        # Add timestamps
        result_df['time_from_start'] = result_df['window'] * window_size
        result_df['exact_time'] = (window_size * result_df['window']) + df['frame.time_relative'].min()

        # Compute direction ratios
        result_df['direction_0_ratio'] = result_df.apply(
            lambda row: (row['packets_count'] - row['most_common_direction']) / row['packets_count'] if row['packets_count'] > 0 else 0,
            axis=1
        )
        result_df['direction_1_ratio'] = result_df.apply(
            lambda row: row['most_common_direction'] / row['packets_count'] if row['packets_count'] > 0 else 0,
            axis=1
        )

        # Limit to 30,000 rows if necessary
        if len(result_df) > 30000:
            print(f"Warning: Output truncated to 30,000 rows from {len(result_df)} rows")
            result_df = result_df.head(30000)

        # Save to CSV
        result_df.to_csv(output_file, index=False)

        # Print summary
        print(f"Processed {len(df)} packets into {len(result_df)} {window_size}-second intervals")

        return result_df

    except Exception as e:
        print(f"Skipping {input_file}: Error encountered - {str(e)}")
        return None

def process_UTMobileNet_traffic(input_file, output_file, window_size):
    """
    Process QUIC traffic data and create an aggregated CSV file with custom window size statistics.
    Skips files that contain NaN values in critical columns.

    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path where the output CSV will be saved
    window_size (float): Size of the time window in seconds (e.g., 0.01 for 10 ms)

    Returns:
    pd.DataFrame or None: The processed dataframe if successful, otherwise None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"\nProcessing file: {input_file}")

        # Ensure required columns exist
        required_columns = ["frame.time", "frame.len", "ip.src", "ip.dst"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Skipping {input_file}: Missing columns {missing_columns}")
            return None

        # Ensure time order

        # Compute 'frame.time_relative' using the first packet as reference
        

        # Extract only the HH:MM:SS.ssssss part
        df["frame.time_relative"] = df["frame.time"].str.extract(r'(\d{2}:\d{2}:\d{2}\.\d+)')
        print(df['frame.time_relative'][:10])

        # Convert to seconds
        df["frame.time_relative"] = pd.to_datetime(df["frame.time_relative"], format="%H:%M:%S.%f").dt.time
        print(df['frame.time_relative'][:10])
        df["frame.time_relative"] = df["frame.time_relative"].apply(lambda t: (t.hour * 3600) + (t.minute * 60) + t.second + t.microsecond / 1e6)
        print(df['frame.time_relative'][:10])
        df = df.sort_values(by='frame.time_relative')  # Ensure time order
        print(df['frame.time_relative'][:10])
        
        # Compute relative time
        start_time = df['frame.time_relative'].iloc[0]
        df['frame.time_relative'] = df['frame.time_relative'] - start_time
        print(df['frame.time_relative'][:10])
        # **Break if there are negative values**
        if (df['frame.time_relative'] < 0).any():
            print("❌ Error: Negative values detected in 'frame.time'. Stopping execution.")
            return None # Stop execution


        # Drop rows where 'frame.time_relative' is NaN
        df = df.dropna(subset=['frame.time_relative'])
        if df.empty:
            print(f"Skipping {input_file}: No valid rows after computing relative time.")
            return None
        #drop row where
        # Compute window index
        df['window'] = (df['frame.time_relative'] // window_size).astype(int)

        # Determine the range of windows
        min_window = df['window'].min()
        max_window = df['window'].max()

        if pd.isna(min_window) or pd.isna(max_window):
            print(f"Skipping {input_file}: Window range calculation resulted in NaN values.")
            return None

        print(f"Window range: {min_window} to {max_window}")

        # Create a DataFrame to hold all possible windows
        all_windows = pd.DataFrame({'window': range(int(min_window), int(max_window) + 1)})

        # Assign a direction based on IP
        print("uniqe src ",len(pd.unique(df['ip.src'])))
        client = df['ip.src'].iloc[0]
        df['Direction'] = df['ip.src'].apply(lambda x: 0 if x == client else 1)

        # Group by window and calculate statistics
        grouped = df.groupby('window').agg({
            'frame.len': ['sum', 'count', 'mean'],
            'Direction': lambda x: Counter(x).most_common(1)[0][0] if not x.empty else np.nan
        }).reset_index()

        # Flatten multi-level columns
        grouped.columns = ['window', 'total_size', 'packets_count', 'average_size', 'most_common_direction']

        # Merge with all_windows to include empty windows
        result_df = pd.merge(all_windows, grouped, on='window', how='left').fillna(0)

        # Add timestamps
        result_df['time_from_start'] = result_df['window'] * window_size
        result_df['exact_time'] = (window_size * result_df['window']) + df['frame.time_relative'].min()

        # Compute direction ratios
        result_df['direction_0_ratio'] = result_df.apply(
            lambda row: (row['packets_count'] - row['most_common_direction']) / row['packets_count'] if row['packets_count'] > 0 else 0,
            axis=1
        )
        result_df['direction_1_ratio'] = result_df.apply(
            lambda row: row['most_common_direction'] / row['packets_count'] if row['packets_count'] > 0 else 0,
            axis=1
        )

        # Limit to 30,000 rows if necessary
        if len(result_df) > 30000:
            print(f"Warning: Output truncated to 30,000 rows from {len(result_df)} rows")
            result_df = result_df.head(30000)

        # Save to CSV
        result_df.to_csv(output_file, index=False)

        # Print summary
        print(f"Processed {len(df)} packets into {len(result_df)} {window_size}-second intervals")

        return result_df

    except Exception as e:
        print(f"Skipping {input_file}: Error encountered - {str(e)}")
        return None

def process_UTMobileNet_traffic_dir(input_file, output_file, window_size):
    """
    Process QUIC traffic data and create an aggregated CSV file with custom window size statistics.
    Skips files that contain NaN values in critical columns.
    the function create fetures as in the Cesnet dataset:
        'window'
        'time_from_start'
        'total_size_dir_-1'
        'packet_count_dir_-1'
        'avg_payload_dir_-1'
        'total_size_dir_1'
        'packet_count_dir_1'
        'avg_payload_dir_1'
        'upstream_downstream_ratio'

    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path where the output CSV will be saved
    window_size (float): Size of the time window in seconds (e.g., 0.01 for 10 ms)

    Returns:
    pd.DataFrame or None: The processed dataframe if successful, otherwise None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        print(f"\nProcessing file: {input_file}")

        # Ensure required columns exist
        required_columns = ["frame.time", "frame.len", "ip.src", "ip.dst"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Skipping {input_file}: Missing columns {missing_columns}")
            return None

        # Extract only the HH:MM:SS.ssssss part
        df["frame.time_relative"] = df["frame.time"].str.extract(r'(\d{2}:\d{2}:\d{2}\.\d+)')
        print(df['frame.time_relative'][:10])

        # Convert to seconds
        df["frame.time_relative"] = pd.to_datetime(df["frame.time_relative"], format="%H:%M:%S.%f").dt.time
        print(df['frame.time_relative'][:10])
        df["frame.time_relative"] = df["frame.time_relative"].apply(lambda t: (t.hour * 3600) + (t.minute * 60) + t.second + t.microsecond / 1e6)
        print(df['frame.time_relative'][:10])
        df = df.sort_values(by='frame.time_relative')  # Ensure time order
        print(df['frame.time_relative'][:10])
        
        # Compute relative time
        start_time = df['frame.time_relative'].iloc[0]
        df['frame.time_relative'] = df['frame.time_relative'] - start_time
        print(df['frame.time_relative'][:10])
        
        # **Break if there are negative values**
        if (df['frame.time_relative'] < 0).any():
            print("❌ Error: Negative values detected in 'frame.time'. Stopping execution.")
            return None # Stop execution

        # Drop rows where 'frame.time_relative' is NaN
        df = df.dropna(subset=['frame.time_relative'])
        if df.empty:
            print(f"Skipping {input_file}: No valid rows after computing relative time.")
            return None
        
        # Compute window index
        df['window'] = (df['frame.time_relative'] // window_size).astype(int)

        # Determine the range of windows
        min_window = df['window'].min()
        max_window = df['window'].max()

        if pd.isna(min_window) or pd.isna(max_window):
            print(f"Skipping {input_file}: Window range calculation resulted in NaN values.")
            return None

        print(f"Window range: {min_window} to {max_window}")

        # Create a DataFrame to hold all possible windows
        all_windows = pd.DataFrame({'window': range(int(min_window), int(max_window) + 1)})

        # Assign direction: -1 for client (source), 1 for server (destination)
        print("unique src ", len(pd.unique(df['ip.src'])))
        client = df['ip.src'].iloc[0]
        df['direction'] = df['ip.src'].apply(lambda x: -1 if x == client else 1)

        # Separate client and server packets
        client_packets = df[df['direction'] == -1]
        server_packets = df[df['direction'] == 1]

        # Group by window for client packets (direction -1)
        client_stats = client_packets.groupby('window').agg({
            'frame.len': ['sum', 'count', 'mean']
        }).reset_index()
        client_stats.columns = ['window', 'total_size_dir_-1', 'packet_count_dir_-1', 'avg_payload_dir_-1']

        # Group by window for server packets (direction 1)
        server_stats = server_packets.groupby('window').agg({
            'frame.len': ['sum', 'count', 'mean']
        }).reset_index()
        server_stats.columns = ['window', 'total_size_dir_1', 'packet_count_dir_1', 'avg_payload_dir_1']

        # Merge all statistics with all_windows
        result_df = all_windows.copy()
        result_df = pd.merge(result_df, client_stats, on='window', how='left')
        result_df = pd.merge(result_df, server_stats, on='window', how='left')

        # Fill NaN values with 0
        result_df = result_df.fillna(0)

        # Calculate upstream/downstream ratio
        # upstream (client) / downstream (server)
        result_df['upstream_downstream_ratio'] = result_df.apply(
            lambda row: row['total_size_dir_-1'] / row['total_size_dir_1'] if row['total_size_dir_1'] > 0 else 0,
            axis=1
        )

        # Add time_from_start
        result_df['time_from_start'] = result_df['window'] * window_size

        # Reorder columns for better readability
        column_order = [
            # 'window',
            'total_size_dir_-1',
            'packet_count_dir_-1',
            'avg_payload_dir_-1',
            'total_size_dir_1',
            'packet_count_dir_1',
            'avg_payload_dir_1',
            'upstream_downstream_ratio',
            'time_from_start'
        ]
        result_df = result_df[column_order]

        # Limit to 30,000 rows if necessary
        if len(result_df) > 30000:
            print(f"Warning: Output truncated to 30,000 rows from {len(result_df)} rows")
            result_df = result_df.head(30000)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        # Save to CSV
        result_df.to_csv(output_file, index=False)

        # Print summary
        print(f"Processed {len(df)} packets into {len(result_df)} {window_size}-second intervals")
        print(f"Client packets: {len(client_packets)}, Server packets: {len(server_packets)}")

        return result_df

    except Exception as e:
        print(f"Skipping {input_file}: Error encountered - {str(e)}")
        return None

def pipeline(input_dir, output_dir, window_size):
    """
    Process all traffic data files in a directory and its subdirectories,
    saving aggregated CSV files in the same folder structure inside output_dir.
    Skips files that contain errors.

    Parameters:
    input_dir (str): Path to the directory containing input CSV files.
    output_dir (str): Path to the directory where output CSV files will be saved.
    window_size (float): Size of the time window in seconds (e.g., 0.01 for 10 ms).
    """

    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        print(f"Directory {input_dir} does not exist.")
        return

    # Iterate over all subdirectories
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)

                # Maintain the same subdirectory structure in the output directory
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                output_file = os.path.join(output_subdir, f"timeseries_{file}")

                # process_VisQuic_traffic(input_file, output_file, window_size)
                # process_TextQuic_traffic(input_file, output_file, window_size)
                # process_UTMobileNet_traffic(input_file, output_file, window_size)
                process_UTMobileNet_traffic_dir(input_file, output_file, window_size)


def create_direction_cesnet_dataset(input_file, output_file, window_size=0.05, big_window_sec=5):
    max_window_limit = int(big_window_sec // window_size)
    start_time = time.time()
    df = pd.read_csv(input_file)
    print(f"Loaded data with shape: {df.shape}")

    with open(output_file, 'w') as f:
        max_windows = []

        for idx, row in df.iterrows():
            try:
                delta_times = json.loads(row['Delta_Time'].replace("'", '"'))
                max_window = min(int(max(np.array(delta_times) / 1000) // window_size), max_window_limit)
                max_windows.append(max_window)
            except Exception as e:
                print(f"Error in row {idx}: {e}")
                max_windows.append(0)

        global_max_window = max(max_windows)
        print(f"Global max window (limited): {global_max_window}")

        # Write header
        header = ['APP']
        dir_values = [-1, 1]
        for i in range(global_max_window + 1):
            for dir in dir_values:
                header.extend([
                    f'total_size_dir_{dir}_window_{i}',
                    f'packet_count_dir_{dir}_window_{i}',
                    f'avg_payload_dir_{dir}_window_{i}'
                ])
            header.append(f'upstream_downstream_ratio_window_{i}')
            header.append(f'time_from_start_{i}')
        f.write(','.join(header) + '\n')

        # Process rows
        for idx, row in df.iterrows():
            try:
                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processing row {idx}/{len(df)} ({elapsed:.1f} sec elapsed)")

                sizes = json.loads(row['SIZE'].replace("'", '"'))
                delta_times = json.loads(row['Delta_Time'].replace("'", '"'))
                dir_values_row = json.loads(row['DIR'].replace("'", '"'))

                windows = {
                    i: {
                        -1: {'total_size': 0, 'packet_count': 0},
                        1: {'total_size': 0, 'packet_count': 0}
                    }
                    for i in range(min(max_windows[idx] + 1, global_max_window + 1))
                }

                for i, dt in enumerate(delta_times):
                    window_idx = int((dt / 1000) // window_size)
                    dir = dir_values_row[i]
                    if dir == 0:
                        continue  # Skip DIR=0 as it means no packet
                    if window_idx in windows and dir in windows[window_idx]:
                        windows[window_idx][dir]['total_size'] += sizes[i]
                        windows[window_idx][dir]['packet_count'] += 1

                row_data = [row['APP']]
                for i in range(global_max_window + 1):
                    us_size = windows[i][1]['total_size'] if i in windows else 0
                    us_count = windows[i][1]['packet_count'] if i in windows else 0
                    ds_size = windows[i][-1]['total_size'] if i in windows else 0
                    ds_count = windows[i][-1]['packet_count'] if i in windows else 0

                    # Upstream stats
                    row_data.append(us_size)
                    row_data.append(us_count)
                    row_data.append(us_size / us_count if us_count > 0 else 0)

                    # Downstream stats
                    row_data.append(ds_size)
                    row_data.append(ds_count)
                    row_data.append(ds_size / ds_count if ds_count > 0 else 0)

                    # Ratio: upstream/downstream size
                    ratio = (us_size / ds_size) if ds_size > 0 else 0
                    row_data.append(ratio)

                    # Time from start
                    row_data.append(i * window_size)

                f.write(','.join(map(str, row_data)) + '\n')

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                f.write(','.join(["0"] * len(header)) + '\n')

    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
    print(f"Output saved to {output_file}")
    return output_file

def create_direction_dataset(input_dirs, output_dir, window_size=5, balance_classes=False):
    """
    Create a dataset from the first window_size timesteps of each sample, organized by direction.
    Saves all samples in two files: features.csv (X) and labels.csv (y)
    
    Parameters:
    input_dirs (list): List of directories containing processed timeseries CSV files (each class in a separate directory)
    output_dir (str): Directory to save the processed dataset
    window_size (int): Number of timesteps to keep from each sample
    balance_classes (bool): Whether to balance dataset by limiting each class to the smallest class size
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return
    counter=0
    # Dictionary to store valid samples per class
    valid_samples = {i: [] for i in range(len(input_dirs))}
    
    # Traverse each class directory recursively
    for class_label, base_dir in enumerate(input_dirs):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.startswith('timeseries_') and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        df = pd.read_csv(file_path)

                        # Ensure time range is sufficient
                        if df["time_from_start"].max() < window_size:
                            print(f"Skipping {file_path}: time range too short")
                            counter+=1
                            continue  # Skip short time-series

                        # Select first window_size seconds of data
                        sample = df[df["time_from_start"] <= window_size]
                        if sample.empty:
                            print(f"Skipping {file_path}: No data in selected time range")
                            counter+=1
                            continue  # Skip empty selections

                        # Store valid (features, label) pairs
                        valid_samples[class_label].append((sample.values.flatten(), class_label))

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    # Determine balancing size after filtering invalid files
    if balance_classes:
        min_class_size = min(len(samples) for samples in valid_samples.values() if samples)
        print(f"Balancing dataset: Using {min_class_size} samples per class")
    else:
        min_class_size = None  # No limit

    # Final lists for dataset
    all_features = []
    all_labels = []

    # Collect balanced samples
    for class_label, samples in valid_samples.items():
        selected_samples = samples if not balance_classes else random.sample(samples, min_class_size)
        
        for feature_vector, label in selected_samples:
            all_features.append(feature_vector)
            all_labels.append(label)

    # Convert lists to DataFrame
    if all_features:
        X = pd.DataFrame(all_features)
        y = pd.Series(all_labels)
        
        # Save combined datasets
        X.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
        y.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

        print(f"⏩ in total skip {counter} samples")
        print(f"✅ Saved {len(all_features)} samples to {output_dir}")
    else:
        print("⚠️ No valid samples found, dataset not created.")

def create_direction_dataset_for_utmobile(input_dirs, output_dir, window_size=5, balance_classes=False):
    """
    Create a dataset from the first window_size timesteps of each sample, organized by direction.
    Saves all samples in two files: features.csv (X) and labels.csv (y)
    
    Parameters:
    input_dirs (list): List of directories containing processed timeseries CSV files (each class in a separate directory)
    output_dir (str): Directory to save the processed dataset
    window_size (int): Number of timesteps to keep from each sample
    balance_classes (bool): Whether to balance dataset by limiting each class to the smallest class size
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return
    
    counter = 0
    global_max_window = -1
    
    # Dictionary to store valid samples per class
    valid_samples = {i: [] for i in range(len(input_dirs))}
    
    # First pass: determine the maximum number of windows across all samples
    for class_label, base_dir in enumerate(input_dirs):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.startswith('timeseries_') and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        df = pd.read_csv(file_path)

                        # Ensure time range is sufficient
                        if df["time_from_start"].max() < window_size:
                            continue

                        # Select first window_size seconds of data
                        sample = df[df["time_from_start"] <= window_size]
                        if sample.empty:
                            continue

                        # Update global max window
                        max_window_in_sample = len(sample) - 1
                        global_max_window = max(global_max_window, max_window_in_sample)

                    except Exception as e:
                        pass
    
    print(f"Global max window: {global_max_window}")
    
    # Generate header based on global_max_window
    dir_values = [-1, 1]
    header = []
    for i in range(global_max_window + 1):
        for dir_val in dir_values:
            header.extend([
                f'total_size_dir_{dir_val}_window_{i}',
                f'packet_count_dir_{dir_val}_window_{i}',
                f'avg_payload_dir_{dir_val}_window_{i}'
            ])
        header.append(f'upstream_downstream_ratio_window_{i}')
        header.append(f'time_from_start_window_{i}')
    
    # Second pass: collect all samples and pad to global_max_window
    for class_label, base_dir in enumerate(input_dirs):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.startswith('timeseries_') and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        df = pd.read_csv(file_path)

                        # Ensure time range is sufficient
                        if df["time_from_start"].max() < window_size:
                            print(f"Skipping {file_path}: time range too short")
                            counter += 1
                            continue

                        # Select first window_size seconds of data
                        sample = df[df["time_from_start"] <= window_size]
                        if sample.empty:
                            print(f"Skipping {file_path}: No data in selected time range")
                            counter += 1
                            continue

                        # Pad sample to match global_max_window
                        num_windows = len(sample)
                        if num_windows < global_max_window + 1:
                            # Pad with zeros
                            padding_rows = global_max_window + 1 - num_windows
                            padding_df = pd.DataFrame(0, index=range(padding_rows), columns=sample.columns)
                            sample = pd.concat([sample, padding_df], ignore_index=True)

                        # Store valid (features, label) pairs
                        valid_samples[class_label].append((sample.values.flatten(), class_label))

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    # Determine balancing size after filtering invalid files
    if balance_classes:
        min_class_size = min(len(samples) for samples in valid_samples.values() if samples)
        print(f"Balancing dataset: Using {min_class_size} samples per class")
    else:
        min_class_size = None

    # Final lists for dataset
    all_features = []
    all_labels = []

    # Collect balanced samples
    for class_label, samples in valid_samples.items():
        selected_samples = samples if not balance_classes else random.sample(samples, min_class_size)
        
        for feature_vector, label in selected_samples:
            all_features.append(feature_vector)
            all_labels.append(label)

    # Convert lists to DataFrame with proper headers
    if all_features:
        X = pd.DataFrame(all_features, columns=header)
        y = pd.Series(all_labels, name='label')
        
        # Save combined datasets
        X.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
        y.to_csv(os.path.join(output_dir, 'labels.csv'), index=False, header=True)

        print(f"⏩ in total skip {counter} samples")
        print(f"✅ Saved {len(all_features)} samples to {output_dir}")
        print(f"Feature shape: {X.shape}")
    else:
        print("⚠️ No valid samples found, dataset not created.")

def delete_pcap_files(directory):
    """
    Deletes all .pcap files from the given directory and its subdirectories.

    Parameters:
    directory (str): Path to the directory where .pcap files should be deleted.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Find all .pcap files recursively
    pcap_files = glob.glob(os.path.join(directory, '**', '*.pcap'), recursive=True)
    key_files = glob.glob(os.path.join(directory, '**', '*.key'), recursive=True)
    # Delete each .pcap file
    for file in pcap_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    for file in key_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

    print(f"\nTotal {len(pcap_files)} .pcap files deleted.")
    print(f"\nTotal {len(key_files)} .key files deleted.")

import os
import shutil

def move_csv_files(source_dir, target_parent_dir):
    """
    Recursively finds all .csv files in source_dir and moves them to target_parent_dir
    while maintaining the original subdirectory structure.

    Parameters:
    source_dir (str): The root directory to search for CSV files.
    target_parent_dir (str): The new parent directory where the files should be moved.
    """

    # Ensure the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Loop through all files in source_dir and its subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.csv'):
                if file.startswith('timeseries_') and file.endswith('.csv'):
                    source_file_path = os.path.join(root, file)

                    # Preserve the relative subdirectory structure
                    relative_path = os.path.relpath(root, source_dir)
                    target_dir = os.path.join(target_parent_dir, relative_path)

                    # Ensure the target directory exists
                    os.makedirs(target_dir, exist_ok=True)

                    # Move the CSV file
                    target_file_path = os.path.join(target_dir, file)
                    shutil.move(source_file_path, target_file_path)

                    print(f"Moved: {source_file_path} -> {target_file_path}")

    print("\n✅ All CSV files have been moved successfully.")

def delete_empty_dirs(directory):
    """
    Recursively deletes all empty directories inside the given directory.

    Parameters:
    directory (str): The root directory to check and delete empty subdirectories.
    """

    # Walk through the directory tree from bottom to top
    for root, dirs, files in os.walk(directory, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)

            # Check if the directory is empty
            if not os.listdir(dir_path):  # Empty directory
                try:
                    os.rmdir(dir_path)
                    print(f"Deleted empty directory: {dir_path}")
                except Exception as e:
                    print(f"Error deleting {dir_path}: {e}")

    # Finally, check and delete the root directory if it's empty
    if not os.listdir(directory):
        try:
            os.rmdir(directory)
            print(f"Deleted root empty directory: {directory}")
        except Exception as e:
            print(f"Error deleting root directory {directory}: {e}")
    else:
        print(f"Directory '{directory}' is not empty.")

# Example usage
if __name__ == "__main__":
    # VisQuic
    # input_files = '../../../data/Chanan/VisQUIC/csv'           
    # output_file_10 = '../../../data/Chanan/VisQUIC/timeseries/10ms'
    # output_file_20 = '../../../data/Chanan/VisQUIC/timeseries/20ms'
    # output_file_30 = '../../../data/Chanan/VisQUIC/timeseries/30ms'
    # output_file_40 = '../../../data/Chanan/VisQUIC/timeseries/40ms'
    # output_file_75 = '../../../data/Chanan/VisQUIC/timeseries/75ms'
    # output_file_100 = '../../../data/Chanan/VisQUIC/timeseries/100ms'
    # output_file_150 = '../../../data/Chanan/VisQUIC/timeseries/150ms'
    # output_file_200 = '../../../data/Chanan/VisQUIC/timeseries/200ms'
    # window_size = 0.01
    # pipeline(input_files, output_file_10, 0.01)
    # pipeline(input_files, output_file_20, 0.02)
    # pipeline(input_files, output_file_30, 0.03)
    # pipeline(input_files, output_file_40, 0.04)
    # pipeline(input_files, output_file_75, 0.075)
    # pipeline(input_files, output_file_100, 0.1)
    # pipeline(input_files, output_file_150, 0.15)
    # pipeline(input_files, output_file_200, 0.2)

    # delete_pcap_files("/home/chanan/Time-Series-Library/VisQUIC/VisQUIC")
    # move_csv_files(source_dir="/home/chanan/Time-Series-Library/VisQUIC/VisQUIC",
    #                target_parent_dir="/home/chanan/Time-Series-Library/VisQUIC/csv")
    # delete_empty_dirs(directory="/home/chanan/Time-Series-Library/VisQUIC/VisQUIC")

    
    # input_dirs_10 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/10ms/youtube.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/10ms/semrush.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/10ms/discord.com"    
    #             ]
    # input_dirs_20 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/20ms/youtube.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/20ms/semrush.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/20ms/discord.com"    
    #             ]
    # input_dirs_30 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/30ms/youtube.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/30ms/semrush.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/30ms/discord.com"    
    #             ]
    # input_dirs_40 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/40ms/youtube.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/40ms/semrush.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/40ms/discord.com"    
    #             ]
    # input_dirs_75 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/75ms/youtube.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/75ms/semrush.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/75ms/discord.com"    
    #             ]
    # input_dirs_100 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/100ms/youtube.com",
    #             "../../../data/Chanan/VisQUIC/timeseries/100ms/semrush.com",
    #             "../../../data/Chanan/VisQUIC/timeseries/100ms/discord.com"
    #             ]
    # input_dirs_150 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/150ms/youtube.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/150ms/semrush.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/150ms/discord.com"    
    #             ]
    # input_dirs_200 = [
    #             "../../../data/Chanan/VisQUIC/timeseries/200ms/youtube.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/200ms/semrush.com", 
    #             "../../../data/Chanan/VisQUIC/timeseries/200ms/discord.com"    
    #             ]
    
    # unbalanced_output_dir_10 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/10ms/1s"
    # unbalanced_output_dir_20 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/20ms/1s"
    # unbalanced_output_dir_30 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/30ms/1s"
    # unbalanced_output_dir_40 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/40ms/1s"
    # unbalanced_output_dir_75 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/75ms/1s"
    # unbalanced_output_dir_100 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/100ms/1s"
    # unbalanced_output_dir_150 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/150ms/1s"
    # unbalanced_output_dir_200 = "../../../data/Chanan/VisQUIC/datasets/unbalanced/200ms/1s"

    # create_direction_dataset(input_dirs_10, unbalanced_output_dir_10, window_size=1, balance_classes=False)
    # create_direction_dataset(input_dirs_20, unbalanced_output_dir_20, window_size=1, balance_classes=False)
    # create_direction_dataset(input_dirs_30, unbalanced_output_dir_30, window_size=1, balance_classes=False)
    # create_direction_dataset(input_dirs_40, unbalanced_output_dir_40, window_size=1, balance_classes=False)
    # create_direction_dataset(input_dirs_75, unbalanced_output_dir_75, window_size=1, balance_classes=False)
    # create_direction_dataset(input_dirs_100, unbalanced_output_dir_100, window_size=1, balance_classes=False)
    # create_direction_dataset(input_dirs_150, unbalanced_output_dir_150, window_size=1, balance_classes=False)
    # create_direction_dataset(input_dirs_200, unbalanced_output_dir_200, window_size=1, balance_classes=False)


    # # Process the data
    # aggregated_df = process_quic_traffic(input_file, output_file, window_size)
    

    # # Display the first few rows of the processed data
    # print("\nFirst few rows of the processed data:")
    # print(aggregated_df.head())

    # input_files0 = ['pretraining/Google Doc/250ms', 'pretraining/Google Drive/250ms','pretraining/Google Music/250ms','pretraining/Google Search/250ms','pretraining/Youtube/250ms']
    # input_files1 = ['pretraining/Google Doc/5ms', 'pretraining/Google Drive/5ms','pretraining/Google Search/5ms','pretraining/Youtube/5ms']
    # input_files2 =  ['pretraining/Google Doc/5ms', 'pretraining/Google Drive/5ms','pretraining/Youtube/5ms']
    # output_file1 = r'C:\Users\shach\Desktop\project FlowPic\big_project\QUIC Dataset\pretraining\Dataset\5ms\10\4 classes (without g.m)'
    # output_file2 = r'C:\Users\shach\Desktop\project FlowPic\big_project\QUIC Dataset\pretraining\Dataset\5ms\10\3 classes (without g.m & g.s)'
    # window_size = 10

    # create_direction_dataset(input_files1, output_file1, window_size)
    # create_direction_dataset(input_files2, output_file2, window_size)

    #Create UTMobileNet DataSet:
    # input_file = "../../../data/Chanan/UTMobileNet/Deterministic Automated Data/youtube/youtube_play-video_2019-03-17_11-49-21_4fd1c357.csv"
    # intput_dirs = ["../../../data/Chanan/UTMobileNet/timeseries/100/youtube",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/twitter",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/spotify",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/reddit",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/pinterest",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/netflix",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/messenger",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/instagram",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/hulu",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/hangout",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/google maps",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/google drive",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/gmail",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/facebook",
    #                "../../../data/Chanan/UTMobileNet/timeseries/100/dropbox",
    #                ]
    # input_dir = "../../../data/Chanan/UTMobileNet/Deterministic Automated Data"
    # output_dir = "../../../data/Chanan/UTMobileNet/dataset/100ms/5s/15 classes"
    # window_size = 0.1
    # # process_UTMobileNet_traffic(input_file=input_file,output_file=output_dir,window_size=window_size)
    # # pipeline(input_dir=input_dir,output_dir=output_dir,window_size=window_size)
    # create_direction_dataset(intput_dirs, output_dir, window_size=5, balance_classes=False)

    # # Create QuicText DataSet:
    # intput_dirs_10 = ["../../../data/Chanan/QuicText/timeseries/10/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/10/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/10/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/10/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/10/Youtube"
    #                  ]
    # intput_dirs_20 = ["../../../data/Chanan/QuicText/timeseries/20/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/20/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/20/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/20/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/20/Youtube"
    #                   ]
    # intput_dirs_30 = ["../../../data/Chanan/QuicText/timeseries/30/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/30/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/30/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/30/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/30/Youtube"
    #                   ]
    # intput_dirs_40 = ["../../../data/Chanan/QuicText/timeseries/40/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/40/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/40/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/40/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/40/Youtube"
    #                   ]
    # intput_dirs_75 = ["../../../data/Chanan/QuicText/timeseries/75/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/75/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/75/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/75/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/75/Youtube"
    #                   ]
    # intput_dirs_100 = ["../../../data/Chanan/QuicText/timeseries/100/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/100/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/100/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/100/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/100/Youtube"
    #                   ]
    # intput_dirs_150 = ["../../../data/Chanan/QuicText/timeseries/150/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/150/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/150/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/150/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/150/Youtube"
    #                   ]
    # intput_dirs_200 = ["../../../data/Chanan/QuicText/timeseries/200/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/200/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/200/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/200/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/200/Youtube"
    #                   ]
    # intput_dirs_250 = ["../../../data/Chanan/QuicText/timeseries/250/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/250/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/250/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/250/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/250/Youtube"
    #                  ]
    # intput_dirs_50 = ["../../../data/Chanan/QuicText/timeseries/50/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/50/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/50/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/50/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/50/Youtube"
    #                  ]
    # intput_dirs_5 = ["../../../data/Chanan/QuicText/timeseries/5/Google Doc",
    #                   "../../../data/Chanan/QuicText/timeseries/5/Google Drive",
    #                   "../../../data/Chanan/QuicText/timeseries/5/Google Music",
    #                   "../../../data/Chanan/QuicText/timeseries/5/Google Search",
    #                   "../../../data/Chanan/QuicText/timeseries/5/Youtube"
    #                  ]
    
    # # # input_dir = "../../..//data/Chanan/TextQuic/csv"
    # output_dir_10 = "../../../data/Chanan/QuicText/datasets/10ms/1s/5 classes"
    # output_dir_20 = "../../../data/Chanan/QuicText/datasets/20ms/1s/5 classes"
    # output_dir_30 = "../../../data/Chanan/QuicText/datasets/30ms/1s/5 classes"
    # output_dir_40 = "../../../data/Chanan/QuicText/datasets/40ms/1s/5 classes"
    # output_dir_75 = "../../../data/Chanan/QuicText/datasets/75ms/1s/5 classes"
    # output_dir_100 = "../../../data/Chanan/QuicText/datasets/100ms/1s/5 classes"
    # output_dir_150 = "../../../data/Chanan/QuicText/datasets/150ms/1s/5 classes"
    # output_dir_200 = "../../../data/Chanan/QuicText/datasets/200ms/1s/5 classes"
    # output_dir_250 = "../../../data/Chanan/QuicText/datasets/250ms/1s/5 classes"
    # output_dir_50 = "../../../data/Chanan/QuicText/datasets/50ms/1s/5 classes"
    # output_dir_5 = "../../../data/Chanan/QuicText/datasets/5ms/1s/5 classes"

    # # intput_dirs = [intput_dirs_20, intput_dirs_30, intput_dirs_40, intput_dirs_75, intput_dirs_100, intput_dirs_150, intput_dirs_200]
    # # output_dirs = [output_dir_20, output_dir_30, output_dir_40, output_dir_75, output_dir_100, output_dir_150, output_dir_200]
    
    # # create_direction_dataset(intput_dirs_10, output_dir_10, window_size=1,balance_classes=False)
    # # create_direction_dataset(intput_dirs_20, output_dir_20, window_size=1,balance_classes=False)
    # # create_direction_dataset(intput_dirs_30, output_dir_30, window_size=1, balance_classes=False)
    # # create_direction_dataset(intput_dirs_40, output_dir_40, window_size=1, balance_classes=False)
    # # create_direction_dataset(intput_dirs_75, output_dir_75, window_size=1, balance_classes=False)
    # # create_direction_dataset(intput_dirs_100, output_dir_100, window_size=1, balance_classes=False)
    # # create_direction_dataset(intput_dirs_150, output_dir_150, window_size=1, balance_classes=False)
    # # create_direction_dataset(intput_dirs_200, output_dir_200, window_size=1, balance_classes=False)
    # create_direction_dataset(intput_dirs_250, output_dir_250, window_size=1, balance_classes=False)
    # create_direction_dataset(intput_dirs_50, output_dir_50, window_size=1, balance_classes=False)
    # create_direction_dataset(intput_dirs_5, output_dir_5, window_size=1, balance_classes=False)

    # # source_dir = "../../../data/Chanan/QuicText/Google Search/50ms"
    # # target_parent_dir = "../../../data/Chanan/QuicText/timeseries/50ms/Google Search"
    # # move_csv_files(source_dir, target_parent_dir)

    # Cesnet DataSet:
    # input_file = "../../../data/Chanan/Cesnet/data/timeseries/dir_data_from_5.csv"
    # output_file200 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_200ms.csv"
    # output_file150 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_150ms.csv"
    # output_file100 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_100ms.csv"
    # output_file75 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_75ms.csv"
    # output_file40 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_40ms.csv"
    # output_file30 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_30ms.csv"
    # output_file20 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_20ms.csv"
    # output_file10 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_10ms.csv"
    # output_file5 = "../../../data/Chanan/Cesnet/data/direction_datasets/dir_cesnet_timeseries_5ms.csv"

    # output_files = [output_file200,
    #            output_file150,
    #            output_file100,
    #            output_file75,
    #            output_file40,
    #            output_file30,
    #            output_file20,
    #            output_file10,
    #            output_file5]
    # window_sizes = [0.2,
    #                 0.15,
    #                 0.1,
    #                 0.075,
    #                 0.04,
    #                 0.03,
    #                 0.02,
    #                 0.01,
    #                 0.005]
    

    # for output_file, window_size in zip(output_files, window_sizes):
    #     create_direction_cesnet_dataset(input_file, output_file, window_size=window_size, big_window_sec=5)
    
#Create UTMobileNet DataSet:
    # input_file = "../../../data/Chanan/UTMobileNet/Deterministic Automated Data/youtube/youtube_play-video_2019-03-17_11-49-21_4fd1c357.csv"
    intput_dirs = ["../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/youtube",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/twitter",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/spotify",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/reddit",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/pinterest",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/netflix",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/messenger",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/instagram",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/hulu",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/hangout",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/google maps",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/google drive",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/gmail",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/facebook",
                   "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20/dropbox",
                   ]
    input_dir = "../../../data/Chanan/UTMobileNet/Deterministic Automated Data"
    output_dir = "../../../data/Chanan/UTMobileNet/timeseries/direction_dataset/20"
    window_size = 0.020
    # process_UTMobileNet_traffic_dir(input_file=input_file,output_file=output_file,window_size=window_size)
    pipeline(input_dir=input_dir,output_dir=output_dir,window_size=window_size)
    output_dir = "../../../data/Chanan/UTMobileNet/direction_dataset/20ms/5s/15 classes"
    create_direction_dataset_for_utmobile(intput_dirs, output_dir, window_size=5, balance_classes=False)
