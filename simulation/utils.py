"""
    File with utility functions
"""

from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, BatchNormalization, Dropout, Reshape, LSTM
from tensorflow.keras.models import Model, load_model
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

"""
    Functions for loading data:
    - google_sheet_to_df()
    - load_processed_frames()
    - load_FM_data()
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

# Load a list with frames_df from the 'processed/' folder
def load_processed_frames(match_id=None, match_ids=None):
    # Create a list for storing all frames DataFrames
    frames_list = []

    # Load frames DataFrames
    for selected_season in seasons:
        for selected_competition in competitions:
            # Define paths
            DATA_FOLDER_PROCESSED = f"{DATA_LOCAL_FOLDER}/data/{selected_season}/{selected_competition}/processed"

            # Find all frames parquet files and extract the available match IDs from the file paths
            match_paths = glob.glob(os.path.join(DATA_FOLDER_PROCESSED, "*.parquet"))
            available_match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths]

            # Determine match_ids to load
            if match_id:
                # Only load the match with the given match_id
                match_ids_to_load = [match_id]
            elif match_ids:
                # Load only the specified match_ids
                match_ids_to_load = match_ids
            else:
                match_ids_to_load = available_match_ids

            # For all matches in match_ids_to_load
            for current_match_id in match_ids_to_load:
                # Convert parquet file to a DataFrame
                file_path_match = os.path.join(DATA_FOLDER_PROCESSED, f"{current_match_id}.parquet")

                if os.path.exists(file_path_match):
                    frames_df = pd.read_parquet(file_path_match)
                    
                    # Append the DataFrame to frames_list
                    frames_list.append(frames_df)

    return frames_list

# Load Football Manager data
def load_FM_data():
    # Load the data from the Google Sheets URL
    fm_players_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ0KwlS1KWQWSNHwpDAyOK5O-0tGC0H6nNapPHNEXGTdmPTBHgDnYm9HyMrdZ79dbLKe1KYDnzOvrno/pub?gid=303039767&single=true&output=csv"
    fm_players_df = google_sheet_to_df(fm_players_url)

    # Manually add the 'ball' row
    ball_data = {
        'Player': ['ball'],
        'Team': ['ball'],
        'Age': [-1],
        'Position': ['ball'],
        'Specific Position': ['ball'],
        'Nationality': ['ball'],
        'Height': [-1.00],
        'Weight': [-1],
        'Acc': [-1],
        'Pac': [-1],
        'Sta': [-1]
    }
    ball_row = pd.DataFrame(ball_data)
    
    # Append the 'ball' row to the DataFrame
    fm_players_df = pd.concat([fm_players_df, ball_row], ignore_index=True)

    return fm_players_df

"""
    Variables used during model training etc
"""

# Define denominators for normalization
denominators = {
    'x': pitch_length,
    'y': pitch_width,
    'v_x': 13,
    'v_y': 13,
    'a_x': 10,
    'a_y': 10,
    'v_x_avg': 13,
    'v_y_avg': 13,
    'acc': 20,
    'pac': 20,
    'sta': 20,
    'height': 2.10,
    'weight': 110,
    'distance_to_ball': round(np.sqrt((pitch_length**2 + pitch_width**2)), 2),
    'angle_to_ball': 360,
    'orientation': 360,
    'tiredness': 10,
    'tiredness_short': 1,
    'minute': 45,
    'period': 2,
}

# Load dataset with Football Manager players
fm_players_df = load_FM_data()

# Define all possible values for each categorical column 
team_directions = ['left', 'right', 'ball']
roles = list(range(-1, 12))
positions = sorted(fm_players_df['Position'].unique())
specific_positions = sorted(fm_players_df['Specific Position'].unique())
nationalities = sorted(fm_players_df['Nationality'].unique())

# Function to calculate embedding dimension with a refined heuristic
def calculate_embedding_dim(input_dim):
    return min(50, max(2, int(np.ceil(input_dim**0.25))))

# Embedding configuration for each categorical column with dynamic output_dim calculation
embedding_config = {
    'team_direction':    {'n_categories': len(team_directions),    'output_dim': calculate_embedding_dim(len(team_directions))},
    'role':              {'n_categories': len(roles),              'output_dim': calculate_embedding_dim(len(roles))},
    'position':          {'n_categories': len(positions),          'output_dim': calculate_embedding_dim(len(positions))},
    'specific_position': {'n_categories': len(specific_positions), 'output_dim': calculate_embedding_dim(len(specific_positions))},
    'nationality':       {'n_categories': len(nationalities),      'output_dim': calculate_embedding_dim(len(nationalities))}
}

# Create mappings
embedding_mapping = {
    'team_direction': {v: i for i, v in enumerate(team_directions)},
    'role': {v: i for i, v in enumerate(roles)},
    'position': {v: i for i, v in enumerate(positions)},
    'specific_position': {v: i for i, v in enumerate(specific_positions)},
    'nationality': {v: i for i, v in enumerate(nationalities)}
}

"""
    General helper functions
    - split_match_ids()
    - euclidean_distance_loss()
    - total_error_loss()
"""

# Split the games into train, test, and validtion. This way, each game will be treated seperatly
def split_match_ids(n_matches):
    # Combined DataFrame for all matches
    all_matches_df = pd.DataFrame()

    # Load match info and concatenate into a single DataFrame
    for selected_season in seasons:
        for selected_competition in competitions:
            match_info_path = f"{DATA_LOCAL_FOLDER}/data/match_info/match_info_{selected_season}_{selected_competition}.parquet"
            match_info_df = pd.read_parquet(match_info_path)
            all_matches_df = pd.concat([all_matches_df, match_info_df], ignore_index=True)

    # Sort the DataFrame to ensure consistent order
    all_matches_df = all_matches_df.sort_values(by='match_id')

    # Fidn the count of each set
    train_count = int(n_matches * train_size)
    test_count = int(n_matches * test_size)
    val_count = n_matches - train_count - test_count

    # Extract the 'match_id' for each set
    train_ids = all_matches_df[all_matches_df['train_test_val'] == 'train']['match_id'].head(train_count).tolist()
    test_ids  = all_matches_df[all_matches_df['train_test_val'] == 'test']['match_id'].head(test_count).tolist()
    val_ids   = all_matches_df[all_matches_df['train_test_val'] == 'val']['match_id'].head(val_count).tolist()

    return train_ids, test_ids, val_ids

# Loss function for model training
def euclidean_distance_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))

# Add a column for distance wrongly predicted (in metres) for each object
def add_pred_error(frames_df):
    # Create a vector with the Eculidian distance between the true position and the predicted position
    frames_df['pred_error'] = round(((frames_df['x_future_pred'] - frames_df[y_cols[0]])**2 + (frames_df['y_future_pred'] - frames_df[y_cols[1]])**2)**0.5, 2)
    
    # Set pred_error to None for rows where 'team_name' is 'ball'
    frames_df.loc[frames_df['team_name'] == 'ball', 'pred_error'] = None
    
    # Set pred_error to None for frames where the ball is not in motion
    frames_df.loc[frames_df['ball_in_motion'] != True, 'pred_error'] = None

    return frames_df

# Add a column for distance wrongly predicted (in metres) for each object. Also return average_pred_error
def total_error_loss(frames_df):
    # Add the 'pred_error' column
    add_pred_error(frames_df)
    
    # Calculate average pred_error, excluding rows where pred_error is None
    average_pred_error = frames_df['pred_error'].mean()

    return round(average_pred_error, 3)

"""
    Functions only for model training
    - add_can_be_sequentialized()
    - prepare_data()
    - get_next_model_filename()
    - adjust_for_embeddings()
    - define_regularizers()
    - prepare_EL_input_data()
    - create_embeddings()
    - prepare_LSTM_df()
    - prepare_LSTM_input_data()
"""

# Add a vector indicating if the row can be sequentialized, i.e. the player has 'sequence_length' consecutive frames
def add_can_be_sequentialized(frames_df, sequence_length, downsampling_factor):
    # Initialize the can_be_sequentialized column to False
    frames_df['can_be_sequentialized'] = False

    # Create temporary vectors with the expected frame for each sequence step
    for i in range(sequence_length):
        frames_df[f'sequence_step_{i}'] = frames_df['frame'] - i * downsampling_factor

    # Group by each unique player
    grouped = frames_df.groupby(['team', 'jersey_number', 'period'])

    # Iterate through each player and find if we can create sequences
    for name, group in grouped:
        # Skip if the group is for the ball. This step is crucial since the function behaves weirdly if we keep the ball frames
        if group['team'].iloc[0] == 'ball':
            continue

        # Convert the frame column to a set for efficient lookups
        frame_set = set(group['frame'])

        # Create temporary columns indicating if each step in the sequences exists
        for i in range(sequence_length):
            frames_df.loc[group.index, f'sequence_step_exists_{i}'] = group[f'sequence_step_{i}'].isin(frame_set)

        # Aggregate 'sequence_step_exists_' checks to set 'can_be_sequentialized'
        sequence_steps_exist_cols = [f'sequence_step_exists_{i}' for i in range(sequence_length)]
        group_can_be_sequentialized = frames_df.loc[group.index, sequence_steps_exist_cols].all(axis=1)

        # Update the main DataFrame
        frames_df.loc[group.index, 'can_be_sequentialized'] = group_can_be_sequentialized

    # Drop temporary columns
    frames_df.drop(columns=[f'sequence_step_{i}' for i in range(sequence_length)], inplace=True)
    frames_df.drop(columns=[f'sequence_step_exists_{i}' for i in range(sequence_length)], inplace=True)

    return frames_df

# Prepare the DataFrame before training
def prepare_df(frames_df, numerical_cols, categorical_cols, positions, downsampling_factor, sequence_length=None):
    # Fill NaN values with zeros for numerical columns
    frames_df[numerical_cols] = frames_df[numerical_cols].fillna(0)

    # Drop rows with NaN values in the labels (y)
    frames_df.dropna(subset=y_cols, inplace=True)

    # Drop rows with NaN values in the numericl_cols
    frames_df.dropna(subset=numerical_cols, inplace=True)

    # TODO: Play around to see if this helps with a large number of games
    # Drop rows where ball is not in motion
    # frames_df = frames_df[frames_df['ball_in_motion']]

    # Only keep ever n:th frame
    frames_df = frames_df[frames_df['frame'] % downsampling_factor == 0]

    # Only keep frames with the specified positions
    frames_df = frames_df[frames_df['position'].isin(positions)]

    # Add the vector 'can_be_sequentialized', if a sequence_length is specified, and column does not exist
    if sequence_length:
        if 'can_be_sequentialized' not in frames_df.columns:
            frames_df = add_can_be_sequentialized(frames_df, sequence_length, downsampling_factor)

    return frames_df

# Prepare and concattened matches, either using a list of 'match_ids' or 'preloaded_frames_df'
def prepare_multiple_dfs(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, positions, downsampling_factor, sequence_length):
    # Initialize lists
    frames_list = []

    # Use the list of match_ids
    if match_ids:
        # For each match
        for match_id in match_ids:
            # Load the frames for the match
            frames_list_match = load_processed_frames(match_id=match_id)
            
            # If we managed to load a DataFrame
            if frames_list_match:
                # Prepare the DataFrame
                current_frames_df = prepare_df(frames_list_match[0].copy(), numerical_cols, categorical_cols, positions, downsampling_factor, sequence_length)

                # Append the data
                frames_list.append(current_frames_df)
    
    # Use the preloaded frames_df
    elif not preloaded_frames_df.empty:
        # Prepare the DataFrame
        current_frames_df = prepare_df(preloaded_frames_df.copy(), numerical_cols, categorical_cols, positions, downsampling_factor, sequence_length)

        # Append the data
        frames_list.append(current_frames_df)
    
    # Throw error since none of the earlier options worked
    else:
        raise ValueError("Either use a list of match_ids or a preloaded frames_df.")

    # Concatenate the lists to create the final DataFrame
    frames_df = pd.concat(frames_list, ignore_index=True)

    return frames_df

# Extract numerical_cols, categorical_cols, and unchanged_cols, then process them
def extract_and_process_columns(frames_df, numerical_cols, categorical_cols, unchanged_cols):
    # Extract the desired columns
    X_data_df = frames_df[numerical_cols + categorical_cols + unchanged_cols].copy()
    y_data_df = frames_df[y_cols].copy()

    # Apply custom label encoding to categorical variables
    for col in categorical_cols:
        # Raise ValueError if we have NaN values in the column
        if X_data_df[col].isnull().any():
            # Print the rows where NaN values are found
            nan_rows = X_data_df[X_data_df[col].isnull()][['position', 'team_name', 'jersey_number', 'match_id']]
            print(f"NaN values found in {col} column in the following rows:\n{nan_rows}")
        
            raise ValueError(f"NaN values found in {col} column. Please handle these before continuing.")

        X_data_df[col] = X_data_df[col].map(embedding_mapping[col])

    # Handling NaNs in numerical columns
    for col in numerical_cols:
        if X_data_df[col].isnull().any():
            X_data_df[col].fillna(X_data_df[col].mean(), inplace=True)  # Fill NaN values with the mean

    # Apply custom normalization
    if normalize:
        for col in numerical_cols:
            if col in denominators:
                X_data_df[col] = X_data_df[col] / denominators[col]

    # Convert categorical columns to int
    X_data_df[categorical_cols] = X_data_df[categorical_cols].astype('int8')

    # Convert numerical columns to float
    X_data_df[numerical_cols] = X_data_df[numerical_cols].astype('float32')

    return X_data_df, y_data_df
    
# Prepare the data, either by usig a list of match_ids or a preloaded frames_df
def prepare_data(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, unchanged_cols, positions, downsampling_factor, sequence_length=None):
    # Prepare matches and concatenate to one DataFrame
    frames_df = prepare_multiple_dfs(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, positions, downsampling_factor, sequence_length)

    # Prepare the desired columns and process them
    X_data_df, y_data_df = extract_and_process_columns(frames_df, numerical_cols, categorical_cols, unchanged_cols)

    return X_data_df, y_data_df

# Get the next model file name based on the number of current models
def get_next_model_filename(model_name):
    models_folder = './models/'

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
def prepare_EL_input_data(match_ids, numerical_cols, categorical_cols, positions, downsampling_factor, preloaded_frames_df=pd.DataFrame()):
    # Prepare data
    X, y = prepare_data(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, [], positions, downsampling_factor)

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

# Sequentialize the numerical and categorical columns
def sequentialize_data(X_df, y_df, numerical_cols, sequence_length, downsampling_factor):
    # Add the y values as a vector
    X_df['y_values'] = y_df.values.tolist()

    # Sort the values on each player from each game
    X_df = X_df.sort_values(by=['team', 'jersey_number', 'period', 'match_id'], ascending=[False, True, True, True])

    # Print error if none of the rows can be sequentialized
    if not X_df['can_be_sequentialized'].any():
        raise ValueError("No frames can be sequentialized based on the provided parameters.")

    # Create lists of all numerical data
    X_df['numerical_data_list'] = X_df[numerical_cols].values.tolist()

    # Create a column for each step in the sequences
    for i in range(sequence_length):
        X_df[f'seq_numerical_{i}'] = X_df['numerical_data_list'].shift(i)

    # Sequentialize the numerical data
    X_df['sequential_numerical_data'] = X_df[[f'seq_numerical_{i}' for i in range(sequence_length)[::-1]]].values.tolist()

    # Clean up temporary columns
    return X_df.drop(columns=[f'seq_numerical_{i}' for i in range(sequence_length)] + ['numerical_data_list'])

# Extract the sequentialized columns
def extract_sequentialized_columns(X_df, categorical_cols):
    # Filter the DataFrame based on can_be_sequentialized
    X_filtered_df = X_df[X_df['can_be_sequentialized']]

    # Extract and prepare numerical sequences
    X_seq_num = np.array(X_filtered_df['sequential_numerical_data'].tolist()).astype('float32')

    # Prepare categorical data
    categorical_inputs = []
    for col in categorical_cols:
        # Reshape data to (None, 1)
        cat_data = np.array(X_filtered_df[col].tolist()).reshape(-1, 1).astype('float32')
        categorical_inputs.append(cat_data)

    # Prepare output data
    y_seq = np.array(X_filtered_df['y_values'].tolist()).astype('float32')

    # Return all inputs as a list, followed by outputs
    return categorical_inputs + [X_seq_num], y_seq

# Create a DataFrame with data ready for LSTM, either by usig a list of match_ids or a preloaded frames_df
def prepare_LSTM_df(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, unchanged_cols, sequence_length, positions, downsampling_factor):
    # Define columns to temporarily give to prepare_data()
    all_unchanged_cols = ['can_be_sequentialized', 'period', 'match_id', 'jersey_number', 'team', 'team_name', 'frame']
    all_unchanged_cols += [col for col in unchanged_cols if col not in all_unchanged_cols]
    all_unchanged_cols = [col for col in all_unchanged_cols if col not in numerical_cols and col not in categorical_cols]

    # Prepare data
    X_df, y_df = prepare_data(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, all_unchanged_cols, positions, downsampling_factor, sequence_length)

    # Sequentialize the data
    X_df = sequentialize_data(X_df, y_df, numerical_cols, sequence_length, downsampling_factor)

    return X_df

# Preapre the LSTM input data
def prepare_LSTM_input_data(match_ids, numerical_cols, categorical_cols, sequence_length, positions, downsampling_factor, preloaded_frames_df=pd.DataFrame()):
    # Prepare the LSTM DataFrame
    X_df = prepare_LSTM_df(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, [], sequence_length, positions, downsampling_factor)

    # Extract the sequentialized columns
    X_seq, y_seq = extract_sequentialized_columns(X_df, categorical_cols)

    return X_seq, y_seq

"""
    Functions for loading, running, and evaluating models:
    - smooth_predictions_xy()
    - run_model()
    - evaluate_model()
    - write_testing_error()
    - test_model()
    - print_column_variance()
"""

# Smooth the vectors 'x_future_pred' and 'y_future_pred'
def smooth_predictions_xy(frames_df, alpha=0.93):
    # Group by unique combinations of 'team_name', 'jersey_number', 'period' and 'match_id'
    grouped = frames_df.groupby(['team_name', 'jersey_number', 'period', 'match_id'])
    
    # Apply the Exponential Moving Average filter to smooth the predictions
    def apply_ema(x):
        return x.ewm(alpha=alpha, adjust=False).mean()

    frames_df['x_future_pred'] = grouped['x_future_pred'].transform(apply_ema)
    frames_df['y_future_pred'] = grouped['y_future_pred'].transform(apply_ema)

    return frames_df

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
    n_matches = 0

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
                elif key == 'matches':
                    n_matches = eval(value)  # Convert string representation to list

    return numerical_cols, categorical_cols, positions, sequence_length, n_matches

# Extract the rows from frames_df containing the ball
def extract_ball_frames(frames_df):
    # Extract the rows containing the ball
    ball_frames_df = frames_df[frames_df['team_name'] == 'ball'].copy()
    frames_df = frames_df[frames_df['team_name'] != 'ball']

    if not ball_frames_df.empty:
        # Set predicted coordinates to 0
        ball_frames_df.loc[:, 'x_future_pred'] = 0
        ball_frames_df.loc[:, 'y_future_pred'] = 0

    return frames_df, ball_frames_df

# Example usage: run_model(test_ids, "NN_model_v1") 
def run_model(match_ids, model_name, downsampling_factor_testing=5, preloaded_frames_df=pd.DataFrame()):
    # Make sure that we either have a list of match_ids, or a preloaded frames_df
    if match_ids == [] and preloaded_frames_df.empty:
        raise ValueError("Either use a list of match_ids or a preloaded frames_df.")

    # Load varibles
    numerical_cols, categorical_cols, positions, sequence_length, _ = extract_variables(model_name)
    positions_with_ball = positions + ['ball']
    
    # Load model
    model = load_tf_model(f'models/{model_name}.h5', euclidean_distance_loss=True)

    # Definie the unchanged columns that we want to keep in 'frames_df'
    unchanged_cols = ['ball_in_motion', 'match_id', 'minute', 'second', 'period', 'frame', 'position', 'team_name'] + y_cols
    unchanged_cols += ['nationality', 'height', 'weight', 'acc', 'pac', 'sta', 'age']

    # Prepare the input data for LSTM model
    if "LSTM" in model_name:
        # Create the DataFrame that will recieve the predictions
        frames_df = prepare_LSTM_df(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, unchanged_cols, sequence_length, positions_with_ball, downsampling_factor_testing)

        # Prepare X_test_input and y_test
        X_test_input, y_test = extract_sequentialized_columns(frames_df, categorical_cols)

        # Clean up temporary columns
        frames_df = frames_df.drop(columns=['y_values', 'sequential_numerical_data'])

        # Extract the rows containing the ball in 'frames_df'
        frames_df, ball_frames_df = extract_ball_frames(frames_df)

    # Prepare the input data for non-LSTM model
    else:
        # Definie columns to temporarely give to prepare_data()
        all_unchanged_cols = [col for col in unchanged_cols if col not in numerical_cols and col not in categorical_cols]

        # Create the DataFrame that will receive the predictions
        frames_df, _ = prepare_data(match_ids, preloaded_frames_df, numerical_cols, categorical_cols, all_unchanged_cols, positions_with_ball, downsampling_factor_testing, sequence_length=None)

        # Extract the rows containing the ball in 'frames_df'
        frames_df, ball_frames_df = extract_ball_frames(frames_df)

        # Prepare X_test_input and y_test
        X_test_input, y_test = prepare_EL_input_data(match_ids, numerical_cols, categorical_cols, positions, downsampling_factor_testing, preloaded_frames_df=preloaded_frames_df)

    # Make predictions using the loaded tf model
    predictions = model.predict(X_test_input)

    # Extract the predicted values
    x_future_pred = predictions[:, 0]
    y_future_pred = predictions[:, 1]

    # Add the predicted values to 'x_future_pred' and 'y_future_pred' columns
    if "LSTM" in model_name:
        # Check that the length of 'x_future_pred' and 'y_future_pred' matches the number of True values in 'can_be_sequentialized'
        assert len(x_future_pred) == len(frames_df[frames_df['can_be_sequentialized']])
        assert len(y_future_pred) == len(frames_df[frames_df['can_be_sequentialized']])

        # Add the predicted values to 'x_future_pred' and 'y_future_pred' columns where 'can_be_sequentialized' is True
        frames_df.loc[frames_df['can_be_sequentialized'], 'x_future_pred'] = x_future_pred
        frames_df.loc[frames_df['can_be_sequentialized'], 'y_future_pred'] = y_future_pred

    else:
        # Check that the length of 'x_future_pred' and 'y_future_pred' matches the length of 'frames_df'
        assert len(x_future_pred) == len(frames_df)
        assert len(y_future_pred) == len(frames_df)

        frames_df['x_future_pred'] = x_future_pred
        frames_df['y_future_pred'] = y_future_pred

    # Unnormalize the numerical columns
    if normalize:
        for col in numerical_cols:
            if col in denominators:
                frames_df[col] = frames_df[col] * denominators[col]

    # Clip values to stay on the pitch
    frames_df['x_future_pred'] = frames_df['x_future_pred'].clip(lower=0, upper=pitch_length)
    frames_df['y_future_pred'] = frames_df['y_future_pred'].clip(lower=0, upper=pitch_width)

    # Smooth the predicted coordinates
    # smooth_predictions_xy(frames_df, alpha=0.98)
    
    # Add the 'ball_frames_df' to 'frames_df'
    frames_df = pd.concat([frames_df, ball_frames_df], ignore_index=True)

    # Sort 'frames_df' on 'match_id' and 'frame'
    frames_df = frames_df.sort_values(by=['frame', 'match_id'])

    return frames_df

# Example usage: evaluate_model(test_ids, "NN_best_v1") 
def evaluate_model(match_ids, model_name, downsampling_factor_testing=1):
    # Run model and the predicted coordinates
    frames_df = run_model(match_ids, model_name, downsampling_factor_testing)
    
    # Calculate the error
    error = total_error_loss(frames_df)

    return error

# Write the testing error to the txt file of the model
def write_testing_error(model_name, error):
    # Define the file path
    file_path = f"models/{model_name}.txt"
    # if 'Testing results' does not exists in txt file
    with open(file_path, 'r') as file:
        if 'Testing results' not in file.read():
            # Write the following with f.write
            with open(file_path, 'a') as file:  # 'a' mode to append data
                file.write(f"\nTesting results:\ntest_loss: {error}\n")
            print("Testing results added to the file.")

# Test an existing model
def test_model(model_name, downsampling_factor_testing=5):
    # Extract n_matches to find the correct test_ids
    _, test_ids, _ = split_match_ids(560)

    # Evaluate the model
    error = evaluate_model(test_ids, model_name, downsampling_factor_testing)
    print(f"Error: {error} m")

    # Write error to txt files
    write_testing_error(model_name, error)

# Disaplys how the average 'pred_error' varies with each value in 'column_to_analyze'
def print_column_variance(match_ids, model_name, column_to_analyze, preloaded_frames_df=pd.DataFrame()):
    # Run model and calculate error
    frames_df = run_model(match_ids, model_name, preloaded_frames_df=preloaded_frames_df)
    error = total_error_loss(frames_df)
    frames_df = add_pred_error(frames_df)

    # Group by 'column_to_analyze' and calculate the average 'pred_error'
    column_variance_df = frames_df.groupby(column_to_analyze)['pred_error'].mean().reset_index()

    # Round to 2 decimal places
    column_variance_df['pred_error'] = round(column_variance_df['pred_error'], 2)

    # Sort by 'column_to_analyze' in ascending order
    column_variance_df = column_variance_df.sort_values(by=column_to_analyze, ascending=True)

    # Print the DataFrame with results
    print(f"Average error: {error}")
    print(f"Average pred error per {column_to_analyze}:")
    print(column_variance_df)
