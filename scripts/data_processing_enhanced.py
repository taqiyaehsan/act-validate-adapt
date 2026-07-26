import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CausalDataProcessor:
    """
    Enhanced data processor for causal discovery with intervention-aware preprocessing.
    Supports multiple datasets and maintains data integrity for causal analysis.
    """
    
    def __init__(self, dataset_type: str = 'smart_room'):
        self.dataset_type = dataset_type
        self.scaling_params = {}
        self.data_statistics = {}
        
        # Define dataset-specific configurations
        self.dataset_configs = {
            'smart_room': {
                'required_columns': ['Temperature', 'Humidity', 'AirQuality', 
                                   'EnergyConsumption', 'OverallSatisfaction'],
                'intervention_vars': ['Temperature', 'Humidity', 'AirQuality'],
                'outcome_vars': ['EnergyConsumption', 'OverallSatisfaction'],
                'normalization_method': 'minmax'
            },
            'smart_building': {
                'required_columns': ['Temperature', 'Humidity', 'AirQuality', 'HVACSetpoint', 
                                   'LightingLevel', 'OccupantCount', 'EnergyConsumption', 
                                   'OverallSatisfaction', 'ThermalComfort', 'VisualComfort',
                                   'AirQualityIndex', 'HVACPower', 'LightingPower'],
                'intervention_vars': ['Temperature', 'Humidity', 'AirQuality', 'HVACSetpoint', 
                                    'LightingLevel', 'OccupantCount'],
                'outcome_vars': ['EnergyConsumption', 'OverallSatisfaction', 'ThermalComfort', 
                               'VisualComfort', 'AirQualityIndex', 'HVACPower', 'LightingPower'],
                'intermediate_vars': ['ThermalComfort', 'VisualComfort', 'AirQualityIndex', 
                                    'HVACPower', 'LightingPower'],
                'normalization_method': 'minmax'
            },
            'physical_smart_room': {
                'required_columns': ['Temperature', 'Humidity', 'AirQuality', 
                                   'EnergyConsumption', 'OverallSatisfaction'],
                'source_columns': ['govee_temp_1', 'govee_temp_2', 'govee_humidity_1', 
                                 'govee_humidity_2', 'bme_iaq', 'kasa_total_energy_kwh', 
                                 'iso7730_satisfaction'],
                'intervention_vars': ['Temperature', 'Humidity', 'AirQuality'],
                'outcome_vars': ['EnergyConsumption', 'OverallSatisfaction'],
                'normalization_method': 'minmax'
            },
            'robot_arm': {
                'required_columns': ['handX', 'handY', 'ballX', 'ballY', 'collision'],
                'intervention_vars': ['handX', 'handY', 'ballX', 'ballY'],
                'outcome_vars': ['collision'],
                'normalization_method': 'coordinate_preserving'
            },
            'ashrae': {
                'required_columns': ['air_temperature', 'dew_temperature', 
                                   'sea_level_pressure', 'meter_reading', 
                                   'square_feet', 'year_built'],
                'intervention_vars': ['air_temperature', 'dew_temperature', 'sea_level_pressure'],
                'outcome_vars': ['meter_reading'],
                'normalization_method': 'robust'
            }
        }
    
    def preprocess_data(self, input_file: str, output_file: str = None, 
                       normalize: bool = True, handle_outliers: bool = True,
                       add_intervention_weights: bool = True) -> pd.DataFrame:
        """
        Comprehensive data preprocessing for causal discovery.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save processed data (optional)
            normalize: Whether to normalize the data
            handle_outliers: Whether to detect and handle outliers
            add_intervention_weights: Whether to add weight column for interventions
            
        Returns:
            Processed DataFrame
        """
        if output_file is None:
            filename, ext = os.path.splitext(input_file)
            output_file = f"{filename}_preprocessed{ext}"
        
        # Load and validate data
        data = self._load_and_validate_data(input_file)
        
        # Clean data
        data = self._clean_data(data)
        
        # Handle outliers if requested
        if handle_outliers:
            data = self._handle_outliers(data)
        
        # Normalize data if requested
        if normalize:
            data = self._normalize_data(data)
        
        # Add intervention weights if requested
        if add_intervention_weights:
            data = self._add_intervention_weights(data)
        
        # Save processed data
        data.to_csv(output_file, index=False)
        
        # Save metadata
        self._save_processing_metadata(output_file)
        
        logger.info(f"Data preprocessing completed. Saved to: {output_file}")
        return data
    
    def _load_and_validate_data(self, input_file: str) -> pd.DataFrame:
        """Load data and perform validation checks."""
        logger.info(f"Loading data from: {input_file}")
        
        try:
            data = pd.read_csv(input_file)
        except Exception as e:
            raise ValueError(f"Error loading data from {input_file}: {str(e)}")
        
        config = self.dataset_configs[self.dataset_type]
        
        # Handle physical smart room data transformation
        if self.dataset_type == 'physical_smart_room':
            data = self._transform_physical_smart_room_data(data, config)
        elif self.dataset_type == 'smart_building':
            data = self._validate_smart_building_data(data, config)
        else:
            # Validate required columns for other dataset types
            required_columns = config['required_columns']
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Select and reorder columns
            data = data[required_columns].copy()
        
        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
        
        return data
    
    def _validate_smart_building_data(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Validate and process smart building simulation data."""
        logger.info("Processing smart building simulation data...")
        
        required_columns = config['required_columns']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            
            # Try to find alternative column names or calculate missing ones
            for col in missing_columns:
                if col == 'ThermalComfort' and 'Temperature' in data.columns:
                    # Calculate thermal comfort from temperature (simplified)
                    data['ThermalComfort'] = np.maximum(0, 100 - np.abs(data['Temperature'] - 22) * 4)
                    logger.info("Calculated ThermalComfort from Temperature")
                elif col == 'VisualComfort' and 'LightingLevel' in data.columns:
                    # Calculate visual comfort from lighting level
                    data['VisualComfort'] = data['LightingLevel'] * 100
                    logger.info("Calculated VisualComfort from LightingLevel")
                elif col == 'AirQualityIndex' and 'AirQuality' in data.columns:
                    # Calculate air quality index (inverted scale)
                    data['AirQualityIndex'] = np.maximum(0, 100 - data['AirQuality'] / 5)
                    logger.info("Calculated AirQualityIndex from AirQuality")
                elif col == 'HVACPower' and 'EnergyConsumption' in data.columns:
                    # Estimate HVAC power as 60% of total energy
                    data['HVACPower'] = data['EnergyConsumption'] * 0.6
                    logger.info("Estimated HVACPower from EnergyConsumption")
                elif col == 'LightingPower' and 'LightingLevel' in data.columns:
                    # Estimate lighting power from lighting level
                    data['LightingPower'] = data['LightingLevel'] * 15
                    logger.info("Estimated LightingPower from LightingLevel")
        
        # Remove timestamp column if present
        if 'Timestamp' in data.columns:
            data = data.drop(columns=['Timestamp'])
        
        # Select available required columns
        available_columns = [col for col in required_columns if col in data.columns]
        data = data[available_columns].copy()
        
        # Store column mapping for reference
        self.column_mapping = {col: [col] for col in available_columns}
        
        logger.info(f"Smart building data processed. Available columns: {available_columns}")
        return data
    
    def _transform_physical_smart_room_data(self, data: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Transform physical smart room data to standard format."""
        logger.info("Transforming physical smart room data...")
        
        # Check for required source columns
        source_columns = config['source_columns']
        missing_columns = [col for col in source_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required source columns: {missing_columns}")
        
        # Create transformed data
        transformed_data = pd.DataFrame()
        
        # Temperature: Average of govee_temp_1 and govee_temp_2
        temp_cols = ['govee_temp_1', 'govee_temp_2']
        temp_data = data[temp_cols].copy()
        
        # Handle missing values in temperature data
        temp_data = temp_data.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate average, handling cases where one sensor might be missing
        transformed_data['Temperature'] = temp_data.mean(axis=1, skipna=True)
        
        # Humidity: Average of govee_humidity_1 and govee_humidity_2
        humidity_cols = ['govee_humidity_1', 'govee_humidity_2']
        humidity_data = data[humidity_cols].copy()
        
        # Handle missing values in humidity data
        humidity_data = humidity_data.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate average, handling cases where one sensor might be missing
        transformed_data['Humidity'] = humidity_data.mean(axis=1, skipna=True)
        
        # AirQuality: BME680 IAQ
        transformed_data['AirQuality'] = data['bme_iaq'].copy()
        
        # EnergyConsumption: Kasa total energy
        transformed_data['EnergyConsumption'] = data['kasa_total_energy_kwh'].copy()
        
        # OverallSatisfaction: ISO 7730 satisfaction
        transformed_data['OverallSatisfaction'] = data['iso7730_satisfaction'].copy()
        
        # Store original column mapping for reference
        self.column_mapping = {
            'Temperature': temp_cols,
            'Humidity': humidity_cols,
            'AirQuality': ['bme_iaq'],
            'EnergyConsumption': ['kasa_total_energy_kwh'],
            'OverallSatisfaction': ['iso7730_satisfaction']
        }
        
        # Log transformation summary
        logger.info("Physical smart room data transformation:")
        logger.info(f"  Temperature: Average of {temp_cols}")
        logger.info(f"  Humidity: Average of {humidity_cols}")
        logger.info(f"  AirQuality: {['bme_iaq']}")
        logger.info(f"  EnergyConsumption: {['kasa_total_energy_kwh']}")
        logger.info(f"  OverallSatisfaction: {['iso7730_satisfaction']}")
        
        return transformed_data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and duplicates."""
        logger.info("Cleaning data...")
        
        # Handle missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
            
            # Use KNN imputation for better results
            imputer = KNNImputer(n_neighbors=5)
            data_imputed = imputer.fit_transform(data)
            data = pd.DataFrame(data_imputed, columns=data.columns, index=data.index)
            
            logger.info("Missing values handled using KNN imputation")
        
        # Remove duplicates
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            logger.info(f"Removing {duplicate_count} duplicate rows")
            data = data.drop_duplicates().reset_index(drop=True)
        
        # Convert data types appropriately
        data = self._optimize_data_types(data)
        
        return data
    
    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency and analysis."""
        config = self.dataset_configs[self.dataset_type]
        
        # Handle boolean columns (like collision in robot arm)
        for col in data.columns:
            if data[col].dtype == bool:
                data[col] = data[col].astype(int)
            elif col in config.get('outcome_vars', []) and data[col].nunique() == 2:
                # Binary outcome variables
                data[col] = data[col].astype(int)
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using robust methods."""
        logger.info("Handling outliers...")
        
        config = self.dataset_configs[self.dataset_type]
        
        # For smart building, be more conservative with outlier detection on satisfaction metrics
        if self.dataset_type == 'smart_building':
            # Don't treat satisfaction metrics as outliers since they have meaningful ranges
            satisfaction_vars = ['OverallSatisfaction', 'ThermalComfort', 'VisualComfort']
            continuous_vars = [col for col in data.columns 
                              if col not in satisfaction_vars
                              and data[col].dtype in ['float64', 'int64']]
        else:
            continuous_vars = [col for col in data.columns 
                              if col not in config.get('outcome_vars', []) 
                              and data[col].dtype in ['float64', 'int64']]
        
        outlier_info = {}
        
        for col in continuous_vars:
            # Use IQR method for outlier detection
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100,
                    'bounds': (lower_bound, upper_bound)
                }
                
                # Cap outliers instead of removing them (preserves causal relationships)
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound
        
        if outlier_info:
            logger.info(f"Handled outliers in {len(outlier_info)} columns")
            for col, info in outlier_info.items():
                logger.info(f"  {col}: {info['count']} outliers ({info['percentage']:.1f}%)")
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data using dataset-appropriate methods."""
        logger.info("Normalizing data...")
        
        config = self.dataset_configs[self.dataset_type]
        method = config['normalization_method']
        
        if method == 'coordinate_preserving':
            return self._coordinate_preserving_normalization(data)
        elif method == 'robust':
            return self._robust_normalization(data)
        else:  # minmax
            return self._minmax_normalization(data)
    
    def _minmax_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standard MinMax normalization to [0,1]."""
        scaler = MinMaxScaler()
        normalized_data = data.copy()
        
        # Apply scaling
        normalized_values = scaler.fit_transform(data)
        normalized_data = pd.DataFrame(normalized_values, columns=data.columns, index=data.index)
        
        # Store scaling parameters
        self.scaling_params = {
            'method': 'minmax',
            'params': {
                'data_min_': scaler.data_min_,
                'data_max_': scaler.data_max_,
                'scale_': scaler.scale_,
                'data_range_': scaler.data_range_
            },
            'feature_names': list(data.columns)
        }
        
        return normalized_data
    
    def _coordinate_preserving_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preserve spatial relationships for coordinate data."""
        normalized_data = data.copy()
        
        # Identify coordinate columns
        coord_columns = [col for col in data.columns if any(axis in col.lower() for axis in ['x', 'y'])]
        other_columns = [col for col in data.columns if col not in coord_columns]
        
        if coord_columns:
            # Global normalization for coordinates
            coord_min = data[coord_columns].min().min()
            coord_max = data[coord_columns].max().max()
            coord_range = coord_max - coord_min
            
            for col in coord_columns:
                normalized_data[col] = (data[col] - coord_min) / coord_range
            
            coord_params = {
                'global_min': coord_min,
                'global_max': coord_max,
                'global_range': coord_range
            }
        else:
            coord_params = {}
        
        # Standard normalization for other columns
        other_params = {}
        if other_columns:
            scaler = MinMaxScaler()
            normalized_data[other_columns] = scaler.fit_transform(data[other_columns])
            
            other_params = {
                'data_min_': scaler.data_min_,
                'data_max_': scaler.data_max_,
                'scale_': scaler.scale_
            }
        
        # Store parameters
        self.scaling_params = {
            'method': 'coordinate_preserving',
            'coord_columns': coord_columns,
            'other_columns': other_columns,
            'coord_params': coord_params,
            'other_params': other_params
        }
        
        return normalized_data
    
    def _robust_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Robust normalization less sensitive to outliers."""
        scaler = RobustScaler()
        
        normalized_values = scaler.fit_transform(data)
        normalized_data = pd.DataFrame(normalized_values, columns=data.columns, index=data.index)
        
        # Store scaling parameters
        self.scaling_params = {
            'method': 'robust',
            'params': {
                'center_': scaler.center_,
                'scale_': scaler.scale_
            },
            'feature_names': list(data.columns)
        }
        
        return normalized_data
    
    def _add_intervention_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add weight column for intervention data handling."""
        # Default weight of 1.0 for observational data
        data['weight'] = 1.0
        return data
    
    def process_intervention_data(self, intervention_results: Dict, 
                                original_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process intervention results and integrate with original data.
        
        Args:
            intervention_results: Dictionary of intervention results
            original_data: Original observational data
            
        Returns:
            Combined dataset with intervention data
        """
        logger.info("Processing intervention data...")
        
        intervention_samples = []
        
        for edge, results in intervention_results.items():
            for result in results:
                if 'post_state' in result or 'postInterventionData' in result:
                    # Extract post-intervention state
                    if 'post_state' in result:
                        post_data = result['post_state']
                    else:
                        post_data = result['postInterventionData']
                    
                    # Create intervention sample
                    sample = {}
                    config = self.dataset_configs[self.dataset_type]
                    
                    for col in config['required_columns']:
                        if col in post_data:
                            sample[col] = float(post_data[col])
                        else:
                            # Use mean from original data as fallback
                            sample[col] = original_data[col].mean()
                    
                    # Higher weight for intervention data
                    sample['weight'] = 2.0
                    sample['intervention_edge'] = f"{edge[0]}->{edge[1]}"
                    
                    intervention_samples.append(sample)
        
        if intervention_samples:
            intervention_df = pd.DataFrame(intervention_samples)
            
            # Combine with original data
            combined_data = pd.concat([original_data, intervention_df], ignore_index=True)
            
            logger.info(f"Added {len(intervention_samples)} intervention samples")
            return combined_data
        
        return original_data
    
    def analyze_causal_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze the causal structure potential of the dataset."""
        config = self.dataset_configs[self.dataset_type]
        
        analysis = {
            'dataset_type': self.dataset_type,
            'total_variables': len(data.columns),
            'intervention_vars': config.get('intervention_vars', []),
            'outcome_vars': config.get('outcome_vars', []),
            'intermediate_vars': config.get('intermediate_vars', []),
            'sample_size': len(data),
            'correlation_strength': {},
            'data_quality': {}
        }
        
        # Correlation analysis between intervention and outcome variables
        intervention_vars = [var for var in config.get('intervention_vars', []) if var in data.columns]
        outcome_vars = [var for var in config.get('outcome_vars', []) if var in data.columns]
        
        for int_var in intervention_vars:
            for out_var in outcome_vars:
                if int_var in data.columns and out_var in data.columns:
                    corr = data[int_var].corr(data[out_var])
                    analysis['correlation_strength'][f"{int_var}->{out_var}"] = abs(corr)
        
        # Data quality metrics
        analysis['data_quality'] = {
            'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100,
            'variance_check': {col: data[col].var() for col in data.select_dtypes(include=[np.number]).columns}
        }
        
        return analysis
    
    def _save_processing_metadata(self, output_file: str):
        """Save processing metadata and parameters."""
        base_name = os.path.splitext(output_file)[0]
        
        # Save scaling parameters as CSV
        if self.scaling_params:
            if self.scaling_params['method'] == 'minmax':
                scaling_df = pd.DataFrame({
                    'feature': self.scaling_params['feature_names'],
                    'data_min': self.scaling_params['params']['data_min_'],
                    'data_max': self.scaling_params['params']['data_max_'],
                    'scale_factor': self.scaling_params['params']['scale_'],
                    'data_range': self.scaling_params['params']['data_range_']
                })
                scaling_df.to_csv(f"{base_name}_scaling_params.csv", index=False)
                logger.info(f"MinMax scaling parameters saved as CSV")
                
            elif self.scaling_params['method'] == 'robust':
                scaling_df = pd.DataFrame({
                    'feature': self.scaling_params['feature_names'],
                    'center': self.scaling_params['params']['center_'],
                    'scale_factor': self.scaling_params['params']['scale_']
                })
                scaling_df.to_csv(f"{base_name}_scaling_params.csv", index=False)
                logger.info(f"Robust scaling parameters saved as CSV")
                
            elif self.scaling_params['method'] == 'coordinate_preserving':
                # Handle coordinate preserving method
                coord_data = []
                other_data = []
                
                # Coordinate columns data
                if self.scaling_params['coord_params']:
                    for col in self.scaling_params['coord_columns']:
                        coord_data.append({
                            'feature': col,
                            'type': 'coordinate',
                            'global_min': self.scaling_params['coord_params']['global_min'],
                            'global_max': self.scaling_params['coord_params']['global_max'],
                            'global_range': self.scaling_params['coord_params']['global_range']
                        })
                
                # Other columns data
                if self.scaling_params['other_params']:
                    for i, col in enumerate(self.scaling_params['other_columns']):
                        other_data.append({
                            'feature': col,
                            'type': 'standard',
                            'data_min': self.scaling_params['other_params']['data_min_'][i],
                            'data_max': self.scaling_params['other_params']['data_max_'][i],
                            'scale_factor': self.scaling_params['other_params']['scale_'][i]
                        })
                
                # Combine and save
                all_scaling_data = coord_data + other_data
                if all_scaling_data:
                    scaling_df = pd.DataFrame(all_scaling_data)
                    scaling_df.to_csv(f"{base_name}_scaling_params.csv", index=False)
                    logger.info(f"Coordinate preserving scaling parameters saved as CSV")
        
        # Save processing report as CSV
        config = self.dataset_configs[self.dataset_type]
        report_data = {
            'dataset_type': [self.dataset_type],
            'processing_timestamp': [pd.Timestamp.now().isoformat()],
            'total_variables': [len(config.get('required_columns', []))],
            'intervention_vars_count': [len(config.get('intervention_vars', []))],
            'outcome_vars_count': [len(config.get('outcome_vars', []))],
            'intermediate_vars_count': [len(config.get('intermediate_vars', []))],
            'normalization_method': [config.get('normalization_method', 'none')],
            'intervention_vars': [', '.join(config.get('intervention_vars', []))],
            'outcome_vars': [', '.join(config.get('outcome_vars', []))],
            'intermediate_vars': [', '.join(config.get('intermediate_vars', []))]
        }
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f"{base_name}_processing_report.csv", index=False)
        
        logger.info(f"Processing metadata saved as CSV files")
    
    def apply_saved_scaling(self, data: pd.DataFrame, scaling_params_file: str) -> pd.DataFrame:
        """Apply previously saved scaling parameters from CSV file to new data."""
        
        # Read scaling parameters from CSV
        scaling_df = pd.read_csv(scaling_params_file)
        normalized_data = data.copy()
        
        # Determine scaling method from the CSV structure
        if 'data_min' in scaling_df.columns and 'scale_factor' in scaling_df.columns:
            # MinMax scaling
            logger.info("Applying MinMax scaling from CSV parameters")
            for _, row in scaling_df.iterrows():
                feature = row['feature']
                if feature in data.columns:
                    data_min = row['data_min']
                    scale_factor = row['scale_factor']
                    normalized_data[feature] = (data[feature] - data_min) * scale_factor
                    
        elif 'center' in scaling_df.columns and 'scale_factor' in scaling_df.columns:
            # Robust scaling
            logger.info("Applying Robust scaling from CSV parameters")
            for _, row in scaling_df.iterrows():
                feature = row['feature']
                if feature in data.columns:
                    center = row['center']
                    scale_factor = row['scale_factor']
                    normalized_data[feature] = (data[feature] - center) / scale_factor
                    
        elif 'type' in scaling_df.columns:
            # Coordinate preserving scaling
            logger.info("Applying Coordinate preserving scaling from CSV parameters")
            coord_rows = scaling_df[scaling_df['type'] == 'coordinate']
            standard_rows = scaling_df[scaling_df['type'] == 'standard']
            
            # Apply coordinate scaling
            if not coord_rows.empty:
                global_min = coord_rows.iloc[0]['global_min']
                global_range = coord_rows.iloc[0]['global_range']
                for _, row in coord_rows.iterrows():
                    feature = row['feature']
                    if feature in data.columns:
                        normalized_data[feature] = (data[feature] - global_min) / global_range
            
            # Apply standard scaling
            for _, row in standard_rows.iterrows():
                feature = row['feature']
                if feature in data.columns:
                    data_min = row['data_min']
                    scale_factor = row['scale_factor']
                    normalized_data[feature] = (data[feature] - data_min) * scale_factor
        
        else:
            logger.warning("Unknown scaling format in CSV file")
            return data
        
        logger.info("Scaling parameters applied successfully from CSV")
        return normalized_data
    
    def generate_data_quality_report(self, data: pd.DataFrame, output_dir: str = None):
        """Generate comprehensive data quality report."""
        if output_dir is None:
            output_dir = os.getcwd()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistical summary
        summary_stats = data.describe()
        summary_stats.to_csv(os.path.join(output_dir, 'data_summary_statistics.csv'))
        
        # Correlation analysis
        plt.figure(figsize=(12, 10))
        correlation_matrix = data.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
        plt.close()
        
        # Distribution plots
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                axes[i].hist(data[col], bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300)
        plt.close()
        
        logger.info(f"Data quality report generated in: {output_dir}")


# Convenience functions for backward compatibility
def preprocess_simulation_data(input_file: str, output_file: str = None, normalize: bool = True) -> pd.DataFrame:
    """Preprocess smart room simulation data."""
    processor = CausalDataProcessor(dataset_type='smart_room')
    return processor.preprocess_data(input_file, output_file, normalize)

def preprocess_smart_building_data(input_file: str, output_file: str = None, normalize: bool = True) -> pd.DataFrame:
    """Preprocess smart building simulation data."""
    processor = CausalDataProcessor(dataset_type='smart_building')
    return processor.preprocess_data(input_file, output_file, normalize)

def preprocess_physical_smart_room_data(input_file: str, output_file: str = None, normalize: bool = True) -> pd.DataFrame:
    """Preprocess physical smart room data with sensor averaging and transformation."""
    processor = CausalDataProcessor(dataset_type='physical_smart_room')
    return processor.preprocess_data(input_file, output_file, normalize)

def preprocess_robot_arm_data(input_file: str, output_file: str = None, 
                             normalize: bool = True, plot_correlations: bool = False) -> pd.DataFrame:
    """Preprocess robot arm data."""
    processor = CausalDataProcessor(dataset_type='robot_arm')
    data = processor.preprocess_data(input_file, output_file, normalize)
    
    if plot_correlations:
        output_dir = os.path.dirname(output_file) if output_file else os.path.dirname(input_file)
        processor.generate_data_quality_report(data, output_dir)
    
    return data

def preprocess_ashrae_data(input_file: str, output_file: str = None, normalize: bool = True) -> pd.DataFrame:
    """Preprocess ASHRAE building energy data."""
    processor = CausalDataProcessor(dataset_type='ashrae')
    return processor.preprocess_data(input_file, output_file, normalize)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Process smart building simulation data
    try:
        # print("Testing Smart Building Data Processing")
        # print("=" * 50)
        
        # # Create sample smart building data
        # sample_data = pd.DataFrame({
        #     'Timestamp': pd.date_range('2025-06-08', periods=1000, freq='5min'),
        #     'Temperature': np.random.normal(22, 3, 1000),
        #     'Humidity': np.random.normal(45, 10, 1000),
        #     'AirQuality': np.random.normal(200, 50, 1000),
        #     'HVACSetpoint': np.random.normal(22, 2, 1000),
        #     'LightingLevel': np.random.uniform(0.3, 1.0, 1000),
        #     'OccupantCount': np.random.poisson(3, 1000),
        #     'EnergyConsumption': np.random.normal(50, 15, 1000),
        #     'OverallSatisfaction': np.random.beta(4, 1, 1000) * 100,
        #     'ThermalComfort': np.random.beta(3, 1, 1000) * 100,
        #     'VisualComfort': np.random.beta(3, 1, 1000) * 100,
        #     'AirQualityIndex': np.random.beta(4, 1, 1000) * 100,
        #     'HVACPower': np.random.normal(30, 10, 1000),
        #     'LightingPower': np.random.normal(15, 5, 1000)
        # })
        
        # # Add some missing values to test robustness
        # sample_data.loc[sample_data.sample(20).index, 'ThermalComfort'] = np.nan
        # sample_data.loc[sample_data.sample(15).index, 'VisualComfort'] = np.nan
        
        # # Save sample data
        # sample_data.to_csv('sample_smart_building_data.csv', index=False)
        # print("✓ Created sample smart building data")
        
        # Process the data using the new function
        processed_data = preprocess_ashrae_data(
            'data/ashrae_data.csv',
            'data/ashrae_data_processed.csv',
            normalize=True
        )
        
        print("✓ ASHRAE data processing completed!")
        # print(f"Original columns: {list(sample_data.columns)}")
        # print(f"Processed columns: {list(processed_data.columns)}")
        # print(f"Processed data shape: {processed_data.shape}")
        
        # Check data ranges after normalization
        # numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        # numeric_cols = [col for col in numeric_cols if col != 'weight']
        # print(f"Processed data range: {processed_data[numeric_cols].min().min():.3f} to {processed_data[numeric_cols].max().max():.3f}")
        
        # # Show sample of transformed data
        # print("\nSample of processed data:")
        # print(processed_data.head())
        
        # # Analyze causal structure
        # processor = CausalDataProcessor(dataset_type='smart_building')
        # causal_analysis = processor.analyze_causal_structure(processed_data)
        # print(f"\nCausal Structure Analysis:")
        # print(f"  Intervention variables: {len(causal_analysis['intervention_vars'])}")
        # print(f"  Outcome variables: {len(causal_analysis['outcome_vars'])}")
        # print(f"  Sample size: {causal_analysis['sample_size']}")
        # print(f"  Strongest correlations:")
        
        # # Show top 5 correlations
        # correlations = causal_analysis['correlation_strength']
        # sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:5]
        # for edge, corr in sorted_corr:
        #     print(f"    {edge}: {corr:.3f}")
        
        # # Generate quality report
        # processor.generate_data_quality_report(processed_data, 'smart_building_quality_report')
        # print("✓ Generated data quality report")
        
        # # Clean up sample files
        # os.remove('sample_smart_building_data.csv')
        # os.remove('sample_smart_building_processed.csv')
        # print("✓ Cleaned up sample files")
        
    except Exception as e:
        print(f"Error in smart building example: {e}")
        import traceback
        traceback.print_exc()
    
    # Example: Process physical smart room data
    # try:
    #     print("\n" + "=" * 50)
    #     print("Testing Physical Smart Room Data Processing")
    #     print("=" * 50)
        
    #     # Create sample physical smart room data
    #     sample_data = pd.DataFrame({
    #         'timestamp': pd.date_range('2025-05-26 17:53:23', periods=1000, freq='30S'),
    #         'heater_on': np.random.choice([True, False], 1000),
    #         'humidifier_on': np.random.choice([True, False], 1000),
    #         'fan_on': np.random.choice([True, False], 1000),
    #         'bme_temperature': np.random.normal(23, 2, 1000),
    #         'bme_humidity': np.random.normal(37, 5, 1000),
    #         'bme_pressure': np.random.normal(1017, 3, 1000),
    #         'bme_gas_resistance': np.random.normal(65000, 5000, 1000),
    #         'bme_iaq': np.random.exponential(20, 1000) + 10,  # IAQ values
    #         'bme_co2_equivalent': np.random.normal(420, 50, 1000),
    #         'govee_temp_1': np.random.normal(20.5, 2, 1000),
    #         'govee_humidity_1': np.random.normal(43.5, 8, 1000),
    #         'govee_temp_2': np.random.normal(24.7, 2, 1000),
    #         'govee_humidity_2': np.random.normal(34.2, 8, 1000),
    #         'kasa_total_energy_kwh': np.random.gamma(0.5, 0.003, 1000),
    #         'kasa_current_power_w': np.random.gamma(2, 274, 1000),
    #         'iso7730_satisfaction': np.random.beta(9, 1, 1000) * 100  # Satisfaction 0-100%
    #     })
        
    #     # Add some missing values to test robustness
    #     sample_data.loc[sample_data.sample(50).index, 'govee_temp_1'] = np.nan
    #     sample_data.loc[sample_data.sample(30).index, 'govee_humidity_2'] = np.nan
        
    #     # Save sample data
    #     sample_data.to_csv('sample_physical_smart_room_data.csv', index=False)
    #     print("✓ Created sample physical smart room data")
        
    #     # Process the data using the function
    #     processed_data = preprocess_physical_smart_room_data(
    #         'sample_physical_smart_room_data.csv',
    #         'sample_physical_smart_room_processed.csv',
    #         normalize=True
    #     )
        
    #     print("✓ Physical smart room data processing completed!")
    #     print(f"Original columns: {list(sample_data.columns)}")
    #     print(f"Processed columns: {list(processed_data.columns)}")
    #     print(f"Processed data shape: {processed_data.shape}")
        
    #     # Show sample of transformed data
    #     print("\nSample of processed data:")
    #     print(processed_data.head())
        
    #     # Clean up sample files
    #     os.remove('sample_physical_smart_room_data.csv')
    #     os.remove('sample_physical_smart_room_processed.csv')
    #     print("✓ Cleaned up sample files")
        
    # except Exception as e:
    #     print(f"Error in physical smart room example: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # print("\n" + "=" * 50)
    # print("Testing Complete!")
    # print("Available preprocessing functions:")
    # print("  - preprocess_smart_building_data() for building simulation CSV files")
    # print("  - preprocess_physical_smart_room_data() for physical sensor CSV files")
    # print("  - preprocess_simulation_data() for basic smart room CSV files")
    # print("  - preprocess_robot_arm_data() for robot arm CSV files")
    # print("  - preprocess_ashrae_data() for ASHRAE building CSV files")