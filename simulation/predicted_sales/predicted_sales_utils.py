"""
    File with utility functions
"""

from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, BatchNormalization, Dropout, Reshape, LSTM
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from tensorflow.keras import regularizers 
from sklearn.pipeline import Pipeline

from io import StringIO
import tensorflow as tf
import pandas as pd
import numpy as np
import requests
import glob
import os

normalize = False

# Define global variables
y_cols = ['Sales Next Month (kr)']

# Define denominators for normalization
denominators = {
    'Price': 1000,
}

# Embedding configuration for each categorical column
embedding_config = {
    'Month': {'n_categories': 12, 'output_dim': 5},
    'Category': {'n_categories': 6, 'output_dim': 3},
    'Machine Group': {'n_categories': 11, 'output_dim': 5},
    'Machine Sub Group': {'n_categories': 35, 'output_dim': 10},
    'Machine Model': {'n_categories': 10, 'output_dim': 5},
    'Product Name': {'n_categories': 2000, 'output_dim': 64}
}

"""
    Functions for loading data:
    - google_sheet_to_df()
"""

# Fetches data from a Google Sheets CSV URL and converts it into a DataFrame
def google_sheet_to_df(url):
    # Fetch the CSV data from the URL
    response = requests.get(url)
    response.raise_for_status()

    # Decode the response text
    response.encoding = 'utf-8'

    # Read the CSV data into a pandas DataFrame
    data = StringIO(response.text)
    data_df = pd.read_csv(data)
    return data_df

"""
    General helper functions
    - split_match_ids()
    - euclidean_distance_loss()
    - total_error_loss()
"""

# Split the games into train, test, and validtion. This way, each game will be treated seperatly
def split_match_ids(match_ids, train_size=0.7, test_size=0.1, val_size=0.2, random_state=42):
    # Calculate the remaining size after the test and validation sizes are removed
    remaining_size = 1.0 - train_size

    # Check if the sum of sizes is not equal to 1
    if remaining_size < 0 or abs(train_size + test_size + val_size - 1.0) > 1e-6:
        raise ValueError("The sum of train_size, test_size, and val_size must be equal to 1.")
    
    # Split the match IDs into train, test, and validation sets
    train_ids, remaining_ids = train_test_split(match_ids, train_size=train_size, random_state=random_state)
    val_ids, test_ids = train_test_split(remaining_ids, test_size=test_size / remaining_size, random_state=random_state)
    
    return train_ids, test_ids, val_ids

# Loss function for model training
def euclidean_distance_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))

# Calculate the error between the Sales and the prediction
def total_error_loss(sales_per_month_df, model_name):
    # Calculate the absolute error between the Sales and the prediction
    sales_per_month_df[f'Error {model_name} (%)'] = round(100 * np.abs(sales_per_month_df['Sales Next Month (kr)'] - sales_per_month_df[f'{model_name} (kr)']) / sales_per_month_df['Sales Next Month (kr)'], 2)

    # Create a copy of the DataFrame
    sales_error_df = sales_per_month_df.copy()

    # Only consider the rows where 'Sales Next Month (kr)' is greater than 100
    sales_error_df = sales_error_df[sales_error_df['Sales Next Month (kr)'] > 100]

    # Drop rows where 'Product' is 'Övrigt'
    sales_error_df = sales_error_df[sales_error_df['Product'] != 'Övrigt']

    # Calculate the mean absolute percentage error
    error = round(sales_error_df[f'Error {model_name} (%)'].mean(), 2)

    return error

"""
    Functions only for model training
    - prepare_data()
    - get_next_model_filename()
    - adjust_for_embeddings()
    - define_regularizers()
    - prepare_EL_input_data()
    - create_embeddings()
    - prepare_LSTM_df()
    - prepare_LSTM_input_data()
"""

# Prepare the DataFrame before training
def prepare_df(sales_per_month_df, numerical_cols, categorical_cols):
    # Fill NaN values with zeros for numerical columns
    sales_per_month_df[numerical_cols] = sales_per_month_df[numerical_cols].fillna(0)

    # Drop rows with NaN values in the labels (y)
    sales_per_month_df.dropna(subset=y_cols, inplace=True)

    return sales_per_month_df

# Prepare data before training
def prepare_data(sales_per_month_df, numerical_cols, categorical_cols, unchanged_cols=[]):
    # Prepate df
    sales_per_month_df = prepare_df(sales_per_month_df.copy(), numerical_cols, categorical_cols)

    # Extract the specified columns
    X_data_df = sales_per_month_df[numerical_cols + categorical_cols + unchanged_cols].copy()
    y_data_df = sales_per_month_df[y_cols].copy()

    # Apply label encoding to categorical variables
    for col in categorical_cols:
        label_encoder = LabelEncoder()
        X_data_df.loc[:, col] = label_encoder.fit_transform(X_data_df[col])

    if normalize:
        # Apply custom normalization
        for col in numerical_cols:
            if col in denominators:
                X_data_df[col] = X_data_df[col] / denominators[col]

    # Convert categorical columns to int
    X_data_df.loc[:, categorical_cols] = X_data_df[categorical_cols].astype('int8')

    # Convert numerical columns to float
    X_data_df.loc[:, numerical_cols] = X_data_df[numerical_cols].astype('float32')

    return X_data_df, y_data_df
    
# Get the next model file name based on the number of current models
def get_next_model_filename(model_name):
    models_folder = "./models/"

    # Get a list of existing model filenames in the models folder
    existing_models = [filename for filename in os.listdir(models_folder) if filename.endswith('.h5') and model_name in filename]

    # Determine the number of existing models
    num_existing_models = len(existing_models)

    # Construct the filename for the next model
    next_model_filename = f"{model_name}_v{num_existing_models + 1}.h5"

    return os.path.join(models_folder, next_model_filename)

# Adjust the X_data for embedding layers
def adjust_for_embeddings(X_data_df, categorical_cols):
    # Split the DataFrame into numerical and categorical components
    X_numerical = X_data_df.drop(columns=categorical_cols)
    
    # Extract and convert categorical columns to a list of arrays if categorical_cols is not empty
    X_categorical = [X_data_df[col].values.reshape(-1, 1) for col in categorical_cols] if categorical_cols else []
    
    return X_numerical, X_categorical

# Choose the appropriate regularizer based on l1 and l2 values.
def define_regularizers(l1=0, l2=0):
    if l1 != 0:
        return regularizers.l1(l1)
    elif l2 != 0:
        return regularizers.l2(l2)
    return None

# Prepare model inputs
def prepare_model_inputs(X_numerical, X_categorical):
    # Prepare input list
    X_input = []

    # Add each categorical column as a separate input if X_categorical is not empty
    if X_categorical:
        X_input.extend(X_categorical)  # This adds each categorical column as a separate input

    # Add the numerical data input after all categorical inputs
    X_input.append(X_numerical)  # Add numerical data as the last input

    return X_input

# Prepare input data for embedding layers
def prepare_EL_input_data(sales_per_month_df, numerical_cols, categorical_cols):
    # Prepare data
    X_data_df, y_data_df = prepare_data(sales_per_month_df, numerical_cols, categorical_cols)

    # No need to do anything more if 'categorical_cols' is empty
    if categorical_cols == []:
        return X_data_df, y_data_df

    # Adjust for embeddings
    X_numerical, X_categorical = adjust_for_embeddings(X_data_df, categorical_cols)

    # Prepare inputs
    X_input = prepare_model_inputs(X_numerical, X_categorical)

    return X_input, y_data_df

# Create an embedding layer for a specific categorical feature.
def create_embedding_layer(name, n_categories, output_dim):
    input_layer = Input(shape=(1,), name=f"{name}_input")  # Input for feature
    embedding = Embedding(input_dim=n_categories, output_dim=output_dim, name=f"{name}_embedding")(input_layer)  # Embedding
    flat_layer = Flatten()(embedding)  # Flatten the output for dense layer
    return input_layer, flat_layer

# Generate embedding layers for all categorical features specified.
def create_embeddings(categorical_cols):
    input_layers = []  # Stores input layers
    flat_layers = []  # Stores flattened outputs
    
    # Process each categorical column
    for col in categorical_cols:
        config = embedding_config.get(col)  # Get config
        if config:
            n_categories = config['n_categories']
            output_dim = config['output_dim']
            input_layer, flat_layer = create_embedding_layer(col, n_categories, output_dim)
            input_layers.append(input_layer)
            flat_layers.append(flat_layer)
    
    return input_layers, flat_layers

# Function to add a bool indicating if the data can be sequentialized
def add_can_be_sequentialized(sales_per_month_df, sequence_length):
    # Determine if the data can be sequentialized
    sales_per_month_df['Can Be Sequentialized'] = sales_per_month_df.groupby(['Machine ID', 'Product'])['Timestep'].shift(sequence_length - 1) == sales_per_month_df['Timestep'] - sequence_length + 1

    return sales_per_month_df

# Sequentialize the numerical and categorical columns
def sequentialize_data(X_df, y_df, numerical_cols, categorical_cols, sequence_length):
    X_df['y_values'] = y_df.values  # Ensure y_df is a Series or properly formatted

    # Print error if none of the rows can be sequentialized
    if not X_df['Can Be Sequentialized'].any():
        raise ValueError("No frames can be sequentialized based on the provided criteria.")

    # Use a more direct approach to handling numerical and categorical data
    X_df['numerical_data_list'] = X_df[numerical_cols].values.tolist()
    
    if categorical_cols:
        X_df['categorical_data_list'] = X_df[categorical_cols].values.tolist()

    # Vectorized shifting and concatenation of data for sequence creation
    for i in range(sequence_length):
        X_df[f'seq_numerical_{i}'] = X_df['numerical_data_list'].shift(i)

    # Assuming 'Can Be Sequentialized' ensures all rows in a sequence are present
    X_df['sequential_numerical_data'] = X_df[[f'seq_numerical_{i}' for i in range(sequence_length)[::-1]]].values.tolist()

    # Clean up temporary columns
    return X_df.drop(columns=[f'seq_numerical_{i}' for i in range(sequence_length)])

# Extract the sequentialized columns
def extract_sequentialized_columns(X_df, categorical_cols):
    X_filtered_df = X_df[X_df['Can Be Sequentialized'] == True]

    # Extract and prepare numerical sequences
    X_seq_num_np = np.array(X_filtered_df['sequential_numerical_data'].tolist()).astype('float32')

    # Prepare categorical data
    categorical_inputs = []
    for col in categorical_cols:
        # Reshape data to (None, 1)
        cat_data = np.array(X_filtered_df[col].tolist()).reshape(-1, 1).astype('float32')
        categorical_inputs.append(cat_data)

    # Prepare output data
    y_seq_np = np.array(X_filtered_df['y_values'].tolist()).astype('float32')

    # Return all inputs as a list, followed by outputs
    return categorical_inputs + [X_seq_num_np], y_seq_np

# Create a DataFrame with data ready for LSTM
def prepare_LSTM_df(sales_per_month_df, numerical_cols, categorical_cols, sequence_length):
    # Add vector 'Can Be Sequentialized'
    sales_per_month_df = add_can_be_sequentialized(sales_per_month_df, sequence_length)

    # Definie columns to temporarely give to prepare_data()
    unchanged_cols = ['Product', 'Machine ID', 'Timestep', 'Can Be Sequentialized']

    # Prepare data
    X_df, y_df = prepare_data(sales_per_month_df, numerical_cols, categorical_cols, unchanged_cols)

    # Sequentialize the data
    X_df = sequentialize_data(X_df, y_df, numerical_cols, categorical_cols, sequence_length)

    return X_df

# Preapre the LSTM input data
def prepare_LSTM_input_data(sales_per_month_df, numerical_cols, categorical_cols, sequence_length):
    # Prepare the LSTM DataFrame
    X_df = prepare_LSTM_df(sales_per_month_df.copy(), numerical_cols, categorical_cols, sequence_length)

    # Extract the sequentialized columns
    X_seq, y_seq = extract_sequentialized_columns(X_df, categorical_cols)

    return X_seq, y_seq

"""
    Functions for loading, running, and evaluating models:
    - smooth_predictions_xy()
    - run_model()
    - evaluate_model()
    - print_column_variance()
"""

# Smooth the vectors 'x_future_pred' and 'y_future_pred'
def smooth_predictions_xy(sales_per_month_df, model_name, alpha=0.93):
    # Group by unique combinations of 'team', 'jersey_number', and 'match_id'
    grouped = sales_per_month_df.groupby(['Machine ID', 'Product', 'Timestep'])
    
    # Apply the Exponential Moving Average filter to smooth the predictions
    def apply_ema(x):
        return x.ewm(alpha=alpha, adjust=False).mean()

    sales_per_month_df[f'{model_name} (kr)'] = grouped[f'{model_name} (kr)'].transform(apply_ema)

    return sales_per_month_df

# Load a tf model
def load_tf_model(model_path, euclidean_distance_loss=False):
    try:
        # Load the model using Keras's load_model function
        if euclidean_distance_loss:
            # Define custom_objects dictionary with the custom loss function
            custom_objects = {'euclidean_distance_loss': euclidean_distance_loss}
            return load_model(model_path, custom_objects=custom_objects) 
        else:
            return load_model(model_path)
    
    except ValueError as e:
        print(e)
        return None

# Extract variable from model txt file
def extract_variables(model_name):
    # Define the file path
    file_path = f"models/{model_name}.txt"

    # Initialize variables to store variables
    numerical_cols = []
    categorical_cols = []
    sequence_length = 0

    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line contains the '=' character
            if '=' in line:
                # Split each line based on '=' and strip whitespace
                key, value = map(str.strip, line.split('='))
                # Check if the line corresponds to numerical_cols or categorical_cols
                if key == 'numerical_cols':
                    numerical_cols = eval(value)  # Convert string representation to list
                elif key == 'categorical_cols':
                    categorical_cols = eval(value)  # Convert string representation to list
                elif key == 'sequence_length':
                    sequence_length = eval(value)  # Convert string representation to list

    return numerical_cols, categorical_cols, sequence_length

# Example usage: run_model(sales_per_month_df, "pSales_NN_v1") 
def run_model(sales_per_month_df, model_name):
    # Load varibles
    numerical_cols, categorical_cols, sequence_length = extract_variables(model_name)

    # Load model
    model = load_tf_model(f"models/{model_name}.h5")

    # Prepared the DataFrames and concatenate into a single large DataFrame
    sales_per_month_df = prepare_df(sales_per_month_df.copy(), numerical_cols, categorical_cols)

    # Prepare the input data for LSTM model
    if "LSTM" in model_name:
        X_test_input, y_test = prepare_LSTM_input_data(sales_per_month_df.copy(), numerical_cols, categorical_cols, sequence_length)

        # Only keep rows that can be sequentialized
        sales_per_month_df = add_can_be_sequentialized(sales_per_month_df, sequence_length)
        sales_per_month_df = sales_per_month_df[sales_per_month_df["Can Be Sequentialized"]]

    # Prepare the input data for non-LSTM model
    else:
        X_test_input, y_test = prepare_EL_input_data(sales_per_month_df, numerical_cols, categorical_cols)

    # Make predictions using the loaded tf model
    predictions = model.predict(X_test_input)

    # Add the predicted values to 'sales_per_month_df'
    sales_per_month_df[f'{model_name} (kr)'] = predictions

    # Smooth the predicted coordinates
    # sales_per_month_df = smooth_predictions_xy(sales_per_month_df, model_name, alpha=0.98)

    return sales_per_month_df

# Example usage: evaluate_model(sales_per_month_df, "pSales_NN_v1") 
def evaluate_model(sales_per_month_df, model_name):
    # Run model and the predicted coordinates
    sales_per_month_df = run_model(sales_per_month_df, model_name)
    
    # Calculate the error
    error = total_error_loss(sales_per_month_df, model_name)

    return error

# Print how the average 'pred_error' varies with each value in 'column_to_analyze'
def print_column_variance(sales_per_month_df, model_name, column_to_analyze):
    # Run model and calculated error
    sales_per_month_df = run_model(sales_per_month_df, model_name)
    error = total_error_loss(sales_per_month_df, model_name)

    # Convert 'pred_error' to numeric, coercing non-numeric values to NaN
    error_col = f'Error {model_name} (%)'
    sales_per_month_df[error_col] = pd.to_numeric(sales_per_month_df[error_col], errors='coerce')

    # Define a function to apply to each group
    def calculate_group_error(group_df):
        # We pass the group_df and model_name to the total_error_loss function
        return total_error_loss(group_df, model_name)
    
    # Group by 'column_to_analyze' and calculate the error for each group
    error_results = sales_per_month_df.groupby(column_to_analyze).apply(calculate_group_error)
    
    # Reset index to turn series into DataFrame and rename the error column
    error_results_df = error_results.reset_index().rename(columns={0: f'Error {model_name} (%)'})
    
    # Round to 2 decimal places
    error_results_df[f'Error {model_name} (%)'] = round(error_results_df[f'Error {model_name} (%)'], 2)
    
    # Sort by 'column_to_analyze' in ascending order
    error_results_df = error_results_df.sort_values(by=column_to_analyze, ascending=True)
    
    # Print the DataFrame with results
    print(f"Average error: {error}")
    print(f"Average pred error per {column_to_analyze}:")
    print(error_results_df)