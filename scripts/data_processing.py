import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_simulation_data(input_file, output_file=None, normalize=True):
    """
    Preprocess the simulation data CSV file by:
    1. Keeping only the required columns
    2. Optionally normalizing the data to range [0,1]
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the output CSV file. 
                                     If None, will use input_file + "_preprocessed.csv"
        normalize (bool): Whether to normalize the data (default: True)
    
    Returns:
        pd.DataFrame: The preprocessed data
    """
    # Set default output filename if not provided
    if output_file is None:
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_preprocessed{ext}"
    
    # Read the CSV file
    data = pd.read_csv(input_file)
    
    # Keep only the required columns
    required_columns = ['Temperature', 'Humidity', 'AirQuality', 
                        'EnergyConsumption', 'OverallSatisfaction']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Select only the required columns
    preprocessed_data = data[required_columns]
    
    # Display original range statistics
    print("Original data ranges:")
    for column in required_columns:
        min_val = preprocessed_data[column].min()
        max_val = preprocessed_data[column].max()
        mean_val = preprocessed_data[column].mean()
        print(f"  {column}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
    
    # Normalize the data if requested
    if normalize:
        # Create a separate DataFrame for the normalized data
        normalized_data = preprocessed_data.copy()
        
        # Initialize MinMaxScaler to scale each feature to [0,1]
        scaler = MinMaxScaler()
        
        # Scale all the columns
        normalized_data[required_columns] = scaler.fit_transform(normalized_data[required_columns])
        
        # Display normalization statistics
        print("\nNormalized data ranges:")
        for column in required_columns:
            min_val = normalized_data[column].min()
            max_val = normalized_data[column].max()
            mean_val = normalized_data[column].mean()
            print(f"  {column}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
        
        # Save the scaling parameters
        scaling_params = {}
        for i, column in enumerate(required_columns):
            scaling_params[column] = {
                'min': scaler.data_min_[i],
                'max': scaler.data_max_[i],
                'scale': scaler.scale_[i],
                'min_scaled': 0,
                'max_scaled': 1
            }
        
        # Create a scaling parameters file
        scaling_filename = os.path.splitext(output_file)[0] + "_scaling_params.csv"
        scaling_df = pd.DataFrame([
            {'column': col, 'original_min': params['min'], 'original_max': params['max'],
             'scale_factor': params['scale']} 
            for col, params in scaling_params.items()
        ])
        scaling_df.to_csv(scaling_filename, index=False)
        print(f"\nScaling parameters saved to {scaling_filename}")
        
        # Use the normalized data for output
        preprocessed_data = normalized_data
    
    # Save the preprocessed data
    preprocessed_data.to_csv(output_file, index=False)
    
    print(f"\nPreprocessed data saved to {output_file}")
    print(f"Original shape: {data.shape}, Preprocessed shape: {preprocessed_data.shape}")
    
    return preprocessed_data

def preprocess_robot_arm_data(input_file, output_file=None, normalize=True, plot_correlations=False):
    """
    Preprocess the robot arm DAG log CSV file by:
    1. Keeping only the required columns
    2. Converting boolean values to integers
    3. Optionally normalizing the data to range [0,1]
    4. Plotting feature distributions and correlations
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the output CSV file. 
                                     If None, will use input_file + "_preprocessed.csv"
        normalize (bool): Whether to normalize the data (default: True)
        plot_correlations (bool): Whether to generate correlation plots (default: False)
    
    Returns:
        pd.DataFrame: The preprocessed data
    """
    # Set default output filename if not provided
    if output_file is None:
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_preprocessed{ext}"
    
    # Read the CSV file
    print(f"Reading data from {input_file}...")
    data = pd.read_csv(input_file)
    
    # Display basic info about the dataset
    print(f"Original dataset shape: {data.shape}")
    print(f"Columns in the dataset: {', '.join(data.columns)}")
    
    # Define required columns for causal analysis
    required_columns = ['handX', 'handY', 'ballX', 'ballY', 'collision']
    
    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Select only the required columns
    preprocessed_data = data[required_columns].copy()
    
    # Convert boolean values to integers (0 and 1)
    if preprocessed_data['collision'].dtype == bool:
        preprocessed_data['collision'] = preprocessed_data['collision'].astype(int)
    
    # Handle missing values
    null_counts = preprocessed_data.isnull().sum()
    if null_counts.sum() > 0:
        print(f"Found missing values: \n{null_counts[null_counts > 0]}")
        # Fill missing values with appropriate strategies
        # For numeric columns use median
        for col in ['handX', 'handY', 'ballX', 'ballY']:
            if null_counts[col] > 0:
                preprocessed_data[col] = preprocessed_data[col].fillna(preprocessed_data[col].median())
        # For collision use most frequent value (mode)
        if null_counts['collision'] > 0:
            preprocessed_data['collision'] = preprocessed_data['collision'].fillna(preprocessed_data['collision'].mode()[0])
        
        print("Missing values have been filled.")
    
    # Remove duplicate rows
    duplicate_count = preprocessed_data.duplicated().sum()
    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows...")
        preprocessed_data = preprocessed_data.drop_duplicates()
    
    # Display original range statistics
    print("\nOriginal data ranges:")
    for column in required_columns:
        min_val = preprocessed_data[column].min()
        max_val = preprocessed_data[column].max()
        mean_val = preprocessed_data[column].mean()
        print(f"  {column}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
    
    # Plot distributions before normalization if requested
    if plot_correlations:
        plot_dir = os.path.dirname(output_file)
        if not os.path.exists(plot_dir) and plot_dir:
            os.makedirs(plot_dir)
            
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = preprocessed_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix Before Normalization')
        correlation_file = os.path.join(plot_dir, 'correlation_matrix.png')
        plt.savefig(correlation_file)
        plt.close()
        print(f"Correlation matrix saved to {correlation_file}")
        
        # Create pairplot for distributions and relationships
        plt.figure(figsize=(15, 15))
        pairplot = sns.pairplot(preprocessed_data, hue='collision', palette='viridis')
        pairplot_file = os.path.join(plot_dir, 'feature_pairplot.png')
        pairplot.savefig(pairplot_file)
        plt.close()
        print(f"Feature pairplot saved to {pairplot_file}")
    
    # Normalize the data if requested
    if normalize:
        # Create a separate DataFrame for the normalized data
        normalized_data = preprocessed_data.copy()
        
        # For coordinates: preserve spatial relationships by using the same scale for X and Y
        coordinate_columns = ['handX', 'handY', 'ballX', 'ballY']
        
        # Find global min and max across all coordinate columns to preserve aspect ratio
        coord_min = preprocessed_data[coordinate_columns].min().min()
        coord_max = preprocessed_data[coordinate_columns].max().max()
        coord_range = coord_max - coord_min
        
        print(f"\nUsing coordinate normalization with global min={coord_min}, max={coord_max}")
        
        # Scale coordinates preserving spatial relationships
        for col in coordinate_columns:
            normalized_data[col] = (preprocessed_data[col] - coord_min) / coord_range
        
        # For non-coordinate columns (collision), use regular normalization
        other_columns = [col for col in required_columns if col not in coordinate_columns]
        
        if other_columns:
            # Initialize MinMaxScaler for non-coordinate columns
            scaler = MinMaxScaler()
            normalized_data[other_columns] = scaler.fit_transform(preprocessed_data[other_columns].values.reshape(-1, len(other_columns)))
        
        # Display normalization statistics
        print("\nNormalized data ranges:")
        for column in required_columns:
            min_val = normalized_data[column].min()
            max_val = normalized_data[column].max()
            mean_val = normalized_data[column].mean()
            print(f"  {column}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
        
        # Save the scaling parameters
        scaling_params = {}
        
        # Save coordinate scaling parameters
        for col in coordinate_columns:
            scaling_params[col] = {
                'min': coord_min,
                'max': coord_max,
                'range': coord_range,
                'type': 'coordinate',
                'min_scaled': 0,
                'max_scaled': 1
            }
        
        # Save non-coordinate scaling parameters
        if other_columns and len(other_columns) > 0:
            for i, col in enumerate(other_columns):
                scaling_params[col] = {
                    'min': scaler.data_min_[i],
                    'max': scaler.data_max_[i],
                    'scale': scaler.scale_[i],
                    'type': 'standard',
                    'min_scaled': 0,
                    'max_scaled': 1
                }
        
        # Create a scaling parameters file
        scaling_filename = os.path.splitext(output_file)[0] + "_scaling_params.csv"
        
        # Prepare scaling data for CSV
        scaling_data = []
        for col, params in scaling_params.items():
            if params['type'] == 'coordinate':
                scaling_data.append({
                    'column': col,
                    'normalization_type': 'coordinate',
                    'global_min': params['min'],
                    'global_max': params['max'],
                    'global_range': params['range']
                })
            else:
                scaling_data.append({
                    'column': col,
                    'normalization_type': 'standard',
                    'original_min': params['min'],
                    'original_max': params['max'],
                    'scale_factor': params['scale']
                })
        
        scaling_df = pd.DataFrame(scaling_data)
        scaling_df.to_csv(scaling_filename, index=False)
        print(f"\nScaling parameters saved to {scaling_filename}")
        
        # Use the normalized data for output
        preprocessed_data = normalized_data
        
        # Plot distributions after normalization if requested
        if plot_correlations:
            # Create correlation heatmap after normalization
            plt.figure(figsize=(10, 8))
            norm_correlation_matrix = preprocessed_data.corr()
            sns.heatmap(norm_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Matrix After Normalization')
            norm_correlation_file = os.path.join(plot_dir, 'normalized_correlation_matrix.png')
            plt.savefig(norm_correlation_file)
            plt.close()
            print(f"Normalized correlation matrix saved to {norm_correlation_file}")
    
    # Add a weight column for PC and SAM
    preprocessed_data['weight'] = 1.0
    
    # Save the preprocessed data
    preprocessed_data.to_csv(output_file, index=False)
    
    print(f"\nPreprocessed data saved to {output_file}")
    print(f"Original shape: {data.shape}, Preprocessed shape: {preprocessed_data.shape}")
    
    return preprocessed_data

def normalize_new_data(data, scaling_params_file):
    """
    Normalize new data using previously saved scaling parameters
    
    Args:
        data (pd.DataFrame): New data to normalize
        scaling_params_file (str): Path to scaling parameters CSV file
    
    Returns:
        pd.DataFrame: Normalized data
    """
    # Read scaling parameters
    scaling_df = pd.read_csv(scaling_params_file)
    
    # Create a copy of the input data
    normalized_data = data.copy()
    
    # Apply scaling to each column
    for _, row in scaling_df.iterrows():
        column = row['column']
        if column in normalized_data.columns:
            original_min = row['original_min']
            original_max = row['original_max']
            scale_factor = row['scale_factor']
            
            # Apply the same scaling transformation
            normalized_data[column] = (normalized_data[column] - original_min) * scale_factor
    
    return normalized_data

# Example usage
if __name__ == "__main__":
    # input_file = "data/simulation_log_2025-03-04T16-42-08-821Z.csv"
    # output_file = "data/simulation_log_2025-03-04T16-42-08-821Z_preprocessed.csv"
    # preprocess_simulation_data(input_file, output_file, normalize=True)

    # input_file = "data/robot_arm_dag_log.csv"
    # output_file = "data/robot_arm_dag_log_preprocessed.csv"
    # preprocess_robot_arm_data(input_file, output_file, normalize=True)

    input_file = "data/smart_room_noisy_raw.csv"
    output_file = "data/smart_room_noisy_preprocessed.csv"
    preprocess_simulation_data(input_file, output_file, normalize=True)