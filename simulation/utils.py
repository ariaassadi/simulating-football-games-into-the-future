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

from settings import *

# Define global variables
y_cols = ['x_future', 'y_future']

# Define denominators for normalization
denominators = {
    'x': pitch_length,
    'y': pitch_width,
    'v_y': 13,
    'v_y': 13,
    'a_y': 10,
    'a_y': 10,
    'acc': 20,
    'pac': 20,
    'sta': 20,
    'height': 2.10,
    'weight': 110,
    'distance_to_ball': round(np.sqrt((pitch_length**2 + pitch_width**2)), 2),
    'angle_to_ball': 360,
    'orientation': 360,
    'tiredness': 10,
    'minute': 45,
    'period': 2,
}

# Embedding configuration for each categorical column
embedding_config = {
    'team_direction': {'n_categories': 2, 'output_dim': 2},
    'role': {'n_categories': 13, 'output_dim': 6},
    'position': {'n_categories': 10, 'output_dim': 5},
    'nationality': {'n_categories': 20, 'output_dim': 7}
}

"""
    Functions for loading data:
    - google_sheet_to_df()
    - load_processed_frames()
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

# Load the processed/frames
def load_processed_frames(n_matches=None, match_id=None):
    # Create DataFrame for storing all frames
    frames_dfs = []
    # Load frames_df
    for selected_season in seasons:
        for selected_competition in competitions:
            # Define paths
            DATA_FOLDER_PROCESSED = f"{DATA_LOCAL_FOLDER}/data/{selected_season}/{selected_competition}/processed"

            # Find all frames parquet files
            match_paths = glob.glob(os.path.join(DATA_FOLDER_PROCESSED, "*.parquet"))

            # Extract the IDs without the ".parquet" extension
            if match_id:
                # Only load the match with the given match_id
                match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths][match_id : match_id + 1]
            elif n_matches:
                # Only load the specified number of matches
                match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths][0:n_matches]
            else:
                # Load all matches
                match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths]

            # For all matches
            for match_id in match_ids:
                # Convert parquet file to a DataFrame
                file_path_match = f"{DATA_FOLDER_PROCESSED}/{match_id}.parquet"
                frames_df = pd.read_parquet(file_path_match)
                
                # Append the DataFrame to frames_dfs
                frames_dfs.append(frames_df)

    return frames_dfs

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

# Add a column for distance wrongly predicted (in metres) for each object
def add_pred_error(frames_df):
    # Create a vector with the Eculidian distance between the true position and the predicted position
    frames_df['pred_error'] = round(((frames_df['x_future_pred'] - frames_df['x_future'])**2 + (frames_df['y_future_pred'] - frames_df['y_future'])**2)**0.5, 2)
    
# Add a column for distance wrongly predicted (in metres) for each object. Also return average_pred_error
def total_error_loss(frames_df, include_ball=False, ball_has_to_be_in_motion=True):
    # Add 'pred_error' column
    add_pred_error(frames_df)
    
    # Create a new column to store modified pred_error values
    frames_df['pred_error_tmp'] = frames_df['pred_error']
    
    # If specified, set pred_error to None for frames where the ball is not in motion
    if ball_has_to_be_in_motion:
        frames_df.loc[frames_df["ball_in_motion"] != True, 'pred_error_tmp'] = None

    # If specified, set pred_error to None for rows where 'team' is 'ball'
    if not include_ball:
        frames_df.loc[frames_df['team'] == 'ball', 'pred_error_tmp'] = None

    # Calculate average pred_error_tmp, excluding rows where pred_error is None
    average_pred_error = frames_df['pred_error_tmp'].mean()

    # Drop the temporary column
    frames_df.drop(columns=['pred_error_tmp'], inplace=True)

    return round(average_pred_error, 3)

"""
    Functions only for model training
    - prepare_data()
    - get_next_model_filename()
    - adjust_for_embeddings()
    - define_regularizers()
    - prepare_EL_input_data()
    - create_embeddings()
    - prepare_LSTM_input_data()
"""

# Prepare the DataFrame before training
def prepare_df(frames_df, numerical_cols, categorical_cols, positions=[], downsampling_factor=downsampling_factor, include_ball=True, ball_has_to_be_in_motion=False):
    # Fill NaN values with zeros for numerical columns
    frames_df[numerical_cols] = frames_df[numerical_cols].fillna(0)

    # Drop rows with NaN values in the labels (y)
    frames_df.dropna(subset=y_cols, inplace=True)

    # Drop rows where 'team' is ball, if specified
    if not include_ball:
        frames_df = frames_df[frames_df['team'] != 'ball']

    # Drop rows where ball is not in motion, if specified
    if ball_has_to_be_in_motion:
        frames_df = frames_df[frames_df['ball_in_motion']]

    # # Drop rows where all objects werent detected
    # frames_df = frames_df[frames_df['objects_tracked'] == 23]

    # Only keep ever n:th frame
    frames_df = frames_df[frames_df['frame'] % downsampling_factor == 0]

    # Only keep frames with the specified positions
    if positions:
        frames_df = frames_df[frames_df['position'].isin(positions)]

    return frames_df

# Prepare data before training
def prepare_data(frames_dfs, numerical_cols, categorical_cols, unchanged_cols=[], positions=[], include_ball=True, ball_has_to_be_in_motion=False):
    # Initialize lists to store features and labels
    X_data = []
    y_data = []

    # For each game
    for frames_df in frames_dfs:
        # Prepare the DataFrame
        frames_df = prepare_df(frames_df, numerical_cols, categorical_cols, positions=positions, downsampling_factor=downsampling_factor, include_ball=True, ball_has_to_be_in_motion=False)

        # Extract features and labels from group
        X = frames_df[numerical_cols + categorical_cols + unchanged_cols]
        y = frames_df[y_cols]

        # Append the data
        X_data.append(X)
        y_data.append(y)

    # Concatenate the lists to create the final feature and label DataFrame
    X_data_df = pd.concat(X_data)
    y_data_df = pd.concat(y_data)

    # Apply label encoding to categorical variables
    for col in categorical_cols:
        label_encoder = LabelEncoder()
        X_data_df[col] = label_encoder.fit_transform(X_data_df[col])

    # Apply custom normalization
    for col in numerical_cols:
        if col in denominators:
            X_data_df[col] = X_data_df[col] / denominators[col]

    # Convert categorical columns to int
    X_data_df[categorical_cols] = X_data_df[categorical_cols].astype('int8')

    # Convert numerical columns to float
    X_data_df[numerical_cols] = X_data_df[numerical_cols].astype('float32')

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
    # Convert list of categorical arrays into a single 2D numpy array if X_categorical is not empty
    X_categorical_concatenated = np.concatenate(X_categorical, axis=1) if X_categorical else None
    
    # Combine numerical and categorical arrays
    X_input = [X_categorical_concatenated, X_numerical] if X_categorical else [X_numerical]
    
    return X_input

# Prepare input data for embedding layers
def prepare_EL_input_data(frames_dfs, numerical_cols, categorical_cols, positions=[]):
    # Prepare data
    X, y = prepare_data(frames_dfs, numerical_cols, categorical_cols, positions=positions, include_ball=False, ball_has_to_be_in_motion=True)

    # No need to do anything more if 'categorical_cols' is empty
    if categorical_cols == []:
        return X, y

    # Adjust for embeddings
    X_numerical, X_categorical = adjust_for_embeddings(X, categorical_cols)

    # Prepare inputs
    X_input = prepare_model_inputs(X_numerical, X_categorical)

    return X_input, y

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

# Add a vector indicating if the row can be sequentialized, i.e. the player has 'sequence_length' consecutive frames
def add_can_be_sequentialized(frames_df, sequence_length):
    # Create temporary vectors with the expexted frame for each sequence step
    for i in range(sequence_length):
        frames_df[f'sequence_step_{i}'] = frames_df['frame'] - i * FPS // downsampling_factor

    # Group by each unique player
    grouped = frames_df.groupby(['team', 'jersey_number', 'match_id'])

    # Iterate through each player and find it we can create sequences
    for _, group in grouped:
        # Convert the frame column to a set for efficient lookups
        frame_set = set(group['frame'])

        # Create temporary columns indicating if each step in the sequences exists
        for i in range(sequence_length):
            group[f'sequence_step_exists_{i}'] = group[f'sequence_step_{i}'].isin(frame_set)

        # Aggregate 'sequence_step_exists_' checks to set 'can_be_sequentialized'
        sequence_steps_exist_cols = [f'sequence_step_exists_{i}' for i in range(sequence_length)]
        group['can_be_sequentialized'] = group[sequence_steps_exist_cols].all(axis=1)

        # Update the main DataFrame
        frames_df.loc[group.index, 'can_be_sequentialized'] = group['can_be_sequentialized']
    
    # Drop temporary columns
    frames_df.drop(columns=[f'sequence_step_{i}' for i in range(sequence_length)], inplace=True)

# Sequentialize the numerical and categorical columns
def sequentialize_data(X_df, y_df, numerical_cols, categorical_cols, sequence_length):
    # Initialize empty lists with sequentialized data
    X_seq_num_data = []
    X_seq_cat_data = []
    y_seq_data = []
    
    # Combined the values in y_df with X_df
    X_df['future_xy'] = y_df.values.tolist()

    # Sort the DataFrame by 'team', 'match_id', and most importantly 'player'
    X_df = X_df.sort_values(by=['team', 'match_id', 'player'])

    # Add vector 'can_be_sequentialized'
    add_can_be_sequentialized(X_df, sequence_length=sequence_length)

    # Create a vector containg a list of all values in the numerical columns
    X_df['numerical_data_list'] = X_df[numerical_cols].values.tolist()
    
    # Create a similar list for the categorical columns, if any
    if categorical_cols:
        # X_df['categorical_data_list'] = X_df[categorical_cols].values.tolist()
        X_df['categorical_data_list'] = X_df[categorical_cols].apply(lambda x: x.tolist(), axis=1)

    # Sort the DataFrame by 'team', 'match_id', and most importantly 'player'
    X_df_sorted = X_df.sort_values(by=['team', 'match_id', 'player'])

    # Group by each unique player
    grouped = X_df_sorted.groupby(['team', 'jersey_number', 'match_id'])

    # Iterate through each player and create sequences
    for _, group in grouped:
        # Create temporary columns with shifted version of 'numerical_cols' and 'categorical_cols'
        for i in range(sequence_length):
            group['numerical_data_list_' + str(i)] = group["numerical_data_list"].shift(i)

        # Concatenate the termporary columns to create the column 'sequential_numerical_data'
        columns_to_sequentialize = ['numerical_data_list_' + str(i) for i in range(sequence_length)][::-1]
        group['sequential_numerical_data'] = group[columns_to_sequentialize].values.tolist()

        # Only consider rows that can be sequentialized
        group = group[group['can_be_sequentialized']]

        # Add the X data to the sequentialized lists
        X_seq_num_data.append(group['sequential_numerical_data'])
        
        if categorical_cols:
            X_seq_cat_data.append(group['categorical_data_list'])

        # Add the y data to the sequentialized lists
        y_seq_data.append(group['future_xy'])

    # Combine all the sequentialized data to create Series
    X_seq_num = pd.concat(X_seq_num_data)
    y_seq = pd.concat(y_seq_data)

    # Convert the Pandas Series of lists to a NumPy array
    X_seq_num_np = np.array(X_seq_num.tolist()).astype('float32')
    y_seq_np = np.array(y_seq.tolist()).astype('float32')
    
    # Add the data from categorical columns to X_seq_np
    if categorical_cols:
        X_seq_cat = pd.concat(X_seq_cat_data)
        X_seq_cat_np = np.array(X_seq_cat.tolist()).astype('float32')
        X_seq_np = [X_seq_cat_np, X_seq_num_np]

        return X_seq_np, y_seq_np
    
    # Return the resuls without adding categorical data
    else:
        return X_seq_num_np, y_seq_np

# Preapre the LSTM input data
def prepare_LSTM_input_data(frames_dfs, numerical_cols, categorical_cols, sequence_length, positions=[]):
    # Definie columns to temporarely give to prepare_data()
    unchanged_cols=['player', 'frame', 'team', 'jersey_number', 'match_id']

    # Prepare data
    X_df, y_df = prepare_data(frames_dfs, numerical_cols=numerical_cols, categorical_cols=categorical_cols, unchanged_cols=unchanged_cols, positions=positions, include_ball=False, ball_has_to_be_in_motion=True)

    # Sequentialize the data
    X_seq, y_seq = sequentialize_data(X_df, y_df, numerical_cols, categorical_cols, sequence_length)

    return X_seq, y_seq

"""
    Functions for loading, running, and evaluating models:
    - smooth_predictions_xy()
    - run_model()
    - evaluate_model()
    - print_column_variance()
"""

# Smooth the vectors 'x_future_pred' and 'y_future_pred'
def smooth_predictions_xy(frames_df, alpha=0.93):
    # Group by unique combinations of 'team', 'jersey_number', and 'match_id'
    grouped = frames_df.groupby(['team', 'jersey_number', 'match_id'])
    
    # Apply the Exponential Moving Average filter to smooth the predictions
    def apply_ema(x):
        return x.ewm(alpha=alpha, adjust=False).mean()

    frames_df['x_future_pred'] = grouped['x_future_pred'].transform(apply_ema)
    frames_df['y_future_pred'] = grouped['y_future_pred'].transform(apply_ema)

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
    positions = []
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
                elif key == 'positions':
                    positions = eval(value)  # Convert string representation to list
                elif key == 'sequence_length':
                    sequence_length = eval(value)  # Convert string representation to list

    return numerical_cols, categorical_cols, positions, sequence_length

# Example usage: run_model(test_frames_dfs, "NN_model_v1") 
def run_model(frames_dfs, model_name):
    # Load varibles
    numerical_cols, categorical_cols, positions, sequence_length = extract_variables(model_name)

    # Load model
    model = load_tf_model(f"models/{model_name}.h5", euclidean_distance_loss=True)

    # Prepared the DataFrames and concatenate into a single large DataFrame
    prepared_frames_dfs = [prepare_df(frames_df, numerical_cols, categorical_cols, positions=positions, downsampling_factor=downsampling_factor) for frames_df in frames_dfs]
    frames_concat_df = pd.concat(prepared_frames_dfs, ignore_index=True)

    # Save the original index before sorting
    original_index = frames_concat_df.index

    # Prepare the input data for LSTM model
    if "LSTM" in model_name:
        X_test_input, y_test = prepare_LSTM_input_data(frames_dfs, numerical_cols, categorical_cols, sequence_length, positions)

        # Only keep rows that can be sequentialized
        add_can_be_sequentialized(frames_concat_df, sequence_length)

        frames_concat_df = frames_concat_df[frames_concat_df["can_be_sequentialized"]]
        print(len(X_test_input))
        print(len(frames_concat_df))

        # Sort the DataFrame by 'team', 'match_id', and most importantly 'player'
        frames_concat_df = frames_concat_df.sort_values(by=['team', 'match_id', 'player'])

    # Prepare the input data for non-LSTM model
    else:
        X_test_input, y_test = prepare_EL_input_data(frames_dfs, numerical_cols, categorical_cols, positions)

    # Make predictions using the loaded tf model
    predictions = model.predict(X_test_input)

    # Extract the predicted values
    x_future_pred = predictions[:, 0]
    y_future_pred = predictions[:, 1]

    # Add the predicted values to 'frames_concat_df'
    frames_concat_df['x_future_pred'] = x_future_pred
    frames_concat_df['y_future_pred'] = y_future_pred

    # Clip values to stay on the pitch
    frames_concat_df['x_future_pred'] = frames_concat_df['x_future_pred'].clip(lower=0, upper=pitch_length)
    frames_concat_df['y_future_pred'] = frames_concat_df['y_future_pred'].clip(lower=0, upper=pitch_width)

    # Smooth the predicted coordinates
    # smooth_predictions_xy(frames_df, alpha=0.98)

    # "Unsort" the DataFrame to get it back to its original order
    frames_concat_df = frames_concat_df.reindex(original_index)

    return frames_concat_df

# Example usage: evaluate_model(test_frames_df, "NN_best_v1") 
def evaluate_model(frames_dfs, model_name):
    # Run model and the predicted coordinates
    frames_concat_df = run_model(frames_dfs, model_name)
    
    # Calculate the error
    error = total_error_loss(frames_concat_df)

    return error

# Disaplys how the average 'pred_error' varies with each value in 'column_to_analyze'
def print_column_variance(frames_dfs, model_name, column_to_analyze):
    # Run model and calculated error
    frames_concat_df = run_model(frames_dfs, model_name)
    error = total_error_loss(frames_concat_df)

    # Convert 'pred_error' to numeric, coercing non-numeric values to NaN
    frames_concat_df['pred_error'] = pd.to_numeric(frames_concat_df['pred_error'], errors='coerce')

    # Group by 'column_to_analyze' and calculate the average 'pred_error'
    column_variance_df = frames_concat_df.groupby(column_to_analyze)['pred_error'].mean().reset_index()

    # Round to 2 decimal places
    column_variance_df['pred_error'] = round(column_variance_df['pred_error'], 2)

    # Sort by 'column_to_analyze' in ascending order
    column_variance_df = column_variance_df.sort_values(by=column_to_analyze, ascending=True)

    # Print the DataFrame with results
    print(f"Average error: {error}")
    print(f"Average pred error per {column_to_analyze}:")
    print(column_variance_df)