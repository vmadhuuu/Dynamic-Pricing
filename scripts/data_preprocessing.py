import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocess_data(input_file='dynamic_pricing.csv', train_file='train_data.csv', test_file='test_data.csv', test_size=0.2, random_state=42):
    """ 
    Load data, perform preprocessing, and split into training and testing datasets.
    """

    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)

    # Construct the correct path to the input file
    input_file_path = os.path.join(current_dir, '..', 'data', input_file)
    train_file_path = os.path.join(current_dir, '..', 'data', train_file)
    test_file_path = os.path.join(current_dir, '..', 'data', test_file)

    # Load the data
    data = pd.read_csv(input_file_path)

    # Identify features and target
    features = data.drop(columns=['Historical_Cost_of_Ride'])
    target = data['Historical_Cost_of_Ride']

    # Preprocessing pipeline for numerical features
    numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Preprocessing pipeline for categorical features
    categorical_features = features.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform the features
    features_preprocessed = preprocessor.fit_transform(features)

    # Combine preprocessed features with target
    preprocessed_data = pd.DataFrame(features_preprocessed)
    preprocessed_data['Historical_Cost_of_Ride'] = target

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(preprocessed_data, test_size=test_size, random_state=random_state)

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)

    # Save the split data for training and evaluation
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)

    print(f"Data preprocessing complete! Training data saved to '{train_file_path}' and testing data saved to '{test_file_path}'.")

if __name__ == "__main__":
    preprocess_data(input_file='dynamic_pricing.csv')
