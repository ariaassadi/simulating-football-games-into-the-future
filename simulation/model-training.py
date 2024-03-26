#!/usr/bin/env python
# coding: utf-8

# # Notebook for training predictive models
# ### Import packages

# In[ ]:


from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.models import load_model
from datetime import date

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import glob
import os

from visualize_game import visualize_training_results
from utils import load_processed_frames, split_match_ids, get_next_model_filename, euclidean_distance_loss, adjust_for_embeddings
from settings import *


# ### Global variables

# In[ ]:


# Define numerical, categorical, and y columns
numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
categorical_cols = ['team_direction', 'role']
y_cols = ['x_future', 'y_future']

# Define parameters for model training
n_epochs = 1
batch_size = 32
n_matches = 40

# Define the length of the sequences
sequence_length = FPS * seconds_into_the_future


# In[ ]:


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
}


# ### Helper functions

# In[ ]:


# Prepare the DataFrame before training
def prepare_df(frames_df, positions=None, include_ball=True, ball_has_to_be_in_motion=False):
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
    n = 5
    frames_df = frames_df[frames_df['frame'] % n == 0]

    # Only keep goalkeeper frames
    if positions:
        frames_df = frames_df[frames_df['position'].isin(positions)]

    return frames_df

# Prepare data before training
def prepare_data(frames_dfs, numerical_cols=numerical_cols, categorical_cols=categorical_cols, positions=None, include_ball=True, ball_has_to_be_in_motion=False):
    # Initialize lists to store features and labels
    X_data = []
    y_data = []

    # For each game
    for frames_df in frames_dfs:
        # Prepare the DataFrame
        frames_df = prepare_df(frames_df, positions=positions, include_ball=True, ball_has_to_be_in_motion=False)

        # Extract features and labels from group
        X = frames_df[numerical_cols + categorical_cols]
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

    return X_data_df, y_data_df

# # TODO: Test this function. It's straight from the goat
# def prepare_sequential_data(X_data, y_data, sequence_length):
#     # Convert pandas DataFrames to NumPy arrays
#     X_data_np = X_data.to_numpy()
#     y_data_np = y_data.to_numpy()

#     data_length = len(X_data)

#     # Create an array of indices to extract sequences
#     indices = np.arange(data_length - sequence_length + 1)[:, None] + np.arange(sequence_length)

#     # Use advanced indexing to extract sequences directly
#     X_seq = X_data_np[indices]
#     y_seq = y_data_np[sequence_length - 1:]

#     return X_seq, y_seq


# In[ ]:


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

# Smooth the vectors 'x_future_pred' and 'y_future_pred'
def smooth_predictions_xy(frames_df, alpha=0.93):
    # Group by unique combinations of 'team', 'jersey_number', and 'match_id'
    grouped = frames_df.groupby(['team', 'jersey_number', 'match_id'])
    
    # Apply the Exponential Moving Average filter to smooth the predictions
    def apply_ema(x):
        return x.ewm(alpha=alpha, adjust=False).mean()

    frames_df['x_future_pred'] = grouped['x_future_pred'].transform(apply_ema)
    frames_df['y_future_pred'] = grouped['y_future_pred'].transform(apply_ema)


# ### Load frames

# In[ ]:


# Load every frames_df to a list
frames_dfs = load_processed_frames(n_matches=n_matches)

# Create an internal match_id for each game
match_ids = range(len(frames_dfs))

# Split match IDs into train, test, and validation sets
train_ids, test_ids, val_ids = split_match_ids(match_ids=match_ids)

# Select frames data for training, testing, and validation
train_frames_dfs = [frames_dfs[i] for i in train_ids]
test_frames_dfs = [frames_dfs[i] for i in test_ids]
val_frames_dfs = [frames_dfs[i] for i in val_ids]


# In[ ]:


# # NAIVE: Always predict that all players will continue with the same velocity and acceleration
# # The calculations are based on x, y, v_x, v_y, a_x, and a_y
# def predict_two_seconds_naive_acceleration(frames_df):
#     # Calculate future positions using kinematic equations
#     frames_df['x_future_naive_acc'] = frames_df['x'] + frames_df['v_x'] * seconds_into_the_future + 0.5 * frames_df['a_x'] * (seconds_into_the_future ** 2)
#     frames_df['y_future_naive_acc'] = frames_df['y'] + frames_df['v_y'] * seconds_into_the_future + 0.5 * frames_df['a_y'] * (seconds_into_the_future ** 2)

# for train_frames_df in train_frames_dfs:
#     predict_two_seconds_naive_acceleration(train_frames_df)

# for val_frames_df in val_frames_dfs:
#     predict_two_seconds_naive_acceleration(val_frames_df)

# for test_frames_df in test_frames_dfs:
#     predict_two_seconds_naive_acceleration(test_frames_df)


# ## Predictive model 1
# ### Embedding layers

# In[ ]:


# Embedding configuration for each categorical column
embedding_config = {
    'team_direction': {'n_categories': 2, 'output_dim': 2},
    'role': {'n_categories': 13, 'output_dim': 6},
    'position': {'n_categories': 10, 'output_dim': 5}
}

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

# Choose the appropriate regularizer based on l1 and l2 values.
def define_regularizers(l1=0, l2=0):
    if l1 != 0:
        return regularizers.l1(l1)
    elif l2 != 0:
        return regularizers.l2(l2)
    return None

# Adjust the X_data for embedding layers
def adjust_for_embeddings(X_data_df, categorical_cols):
    # Split the DataFrame into numerical and categorical components
    X_numerical = X_data_df.drop(columns=categorical_cols)
    
    # Extract and convert categorical columns to a list of arrays if categorical_cols is not empty
    X_categorical = [X_data_df[col].values.reshape(-1, 1) for col in categorical_cols] if categorical_cols else []
    
    return X_numerical, X_categorical

def prepare_model_inputs(X_numerical, X_categorical):
    # Convert list of categorical arrays into a single 2D numpy array if X_categorical is not empty
    X_categorical_concatenated = np.concatenate(X_categorical, axis=1) if X_categorical else None
    
    # Combine numerical and categorical arrays
    X_input = [X_categorical_concatenated, X_numerical] if X_categorical else [X_numerical]
    
    return X_input

def prepare_EL_input_data(frames_dfs, numerical_cols, categorical_cols, positions=None):
    # Prepare data
    X, y = prepare_data(frames_dfs, numerical_cols=numerical_cols, categorical_cols=categorical_cols, positions=positions, include_ball=False, ball_has_to_be_in_motion=True)

    # No need to do anything more if 'categorical_cols' is empty
    if categorical_cols == []:
        return X, y

    # Adjust for embeddings
    X_numerical, X_categorical = adjust_for_embeddings(X, categorical_cols)

    # Prepare inputs
    X_input = prepare_model_inputs(X_numerical, X_categorical)

    return X_input, y


# In[ ]:


# Define the neural network model with embeddings for categorical features.
def define_NN_model_with_embedding(numerical_input_shape, l1=0, l2=0):
    if categorical_cols:
        categorical_inputs, categorical_flats = create_embeddings(categorical_cols)  # Create embeddings
        numerical_input = Input(shape=(numerical_input_shape,), name='numerical_input')  # Numerical input
        concatenated_features = Concatenate()([*categorical_flats, numerical_input])  # Combine all features
        model_inputs = [*categorical_inputs, numerical_input]  # Model inputs
    else:
        numerical_input = Input(shape=(numerical_input_shape,), name='numerical_input')  # Numerical input
        concatenated_features = numerical_input  # Use only numerical input
        model_inputs = numerical_input  # Model inputs
    
    # Dense layers
    regularizer = define_regularizers(l1, l2)  # Set regularizer
    dense_layer_1 = Dense(64, activation='relu', kernel_regularizer=regularizer)(concatenated_features)
    dense_layer_2 = Dense(32, activation='relu', kernel_regularizer=regularizer)(dense_layer_1)

    output_layer = Dense(2)(dense_layer_2)  # Output layer for x_future and y_future
    model = Model(inputs=model_inputs, outputs=output_layer)  # Build model

    return model

def train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=None, l1=0, l2=0, special_text=None):
    # Prepare inputs
    X_train_input, y_train = prepare_EL_input_data(train_frames_dfs, numerical_cols, categorical_cols, positions=positions)
    X_val_input, y_val = prepare_EL_input_data(val_frames_dfs, numerical_cols, categorical_cols, positions=positions)

    # Define the model
    model = define_NN_model_with_embedding(numerical_input_shape=len(numerical_cols), l1=l1, l2=l2)

    # Compile the model
    model.compile(optimizer='adam', loss=euclidean_distance_loss)

    # Train the model with the corrected input format
    history = model.fit(X_train_input, y_train, validation_data=(X_val_input, y_val), epochs=n_epochs, batch_size=batch_size, verbose=2)

    # Save the trained model to disk
    model_filename = get_next_model_filename("NN_model")
    model.save(model_filename)

    # Generate the corresponding txt filename
    output_txt_filename = os.path.splitext(model_filename)[0] + ".txt"

    # Write the output directly to the txt file
    with open(output_txt_filename, 'w') as f:
        # Write the some general info at the begging of the file
        today_date = date.today().strftime("%Y-%m-%d")
        f.write(f"{today_date}\n")
        f.write(f"epochs={n_epochs}\n")
        f.write(f"matches={n_matches}\n")
        f.write(f"numerical_cols={numerical_cols}\n")
        f.write(f"categorical_cols={categorical_cols}\n")
        f.write(f"positions={positions}\n")
        if l1 != 0: f.write(f"l1={l1}\n")
        if l2 != 0: f.write(f"l2={l2}\n")
        if special_text: f.write(f"{special_text}\n")

        # Write the training results
        f.write("\nTraining results:\n")
        for key, value in history.history.items():
            rounded_values = [round(v, 2) for v in value]
            f.write(f"{key}: {rounded_values}\n")


# In[ ]:


# Train the NN model with embedding layers
n_epochs = 1
categorical_cols = ["position"]
positions = ["Attacking Midfielder", "Central Midfielder", "Centre-Back", "Defensive Midfielder", "Forward", "Full-Back", "Goalkeeper", "Wide Midfielder", "Winger"]
# positions = ["Goalkeeper"]

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness']
# train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)


# In[ ]:


n_epochs = 5
categorical_cols = []
positions = ["Attacking Midfielder", "Central Midfielder", "Centre-Back", "Defensive Midfielder", "Forward", "Full-Back", "Goalkeeper", "Wide Midfielder", "Winger"]

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'pac', 'acc']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'height', 'weight', 'pac', 'acc']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'height']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'weight']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'height', 'weight']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions)

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball']
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions, l2=0.00001)
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions, l2=0.0001)
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions, l2=0.001)
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions, l2=0.01)
train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, positions=positions, l2=0.1)


# In[ ]:


# Load a tf model
def load_model(model_path, euclidean_distance_loss=False):
    try:
        # Load the model using Keras's load_model function
        if euclidean_distance_loss:
            # Define custom_objects dictionary with the custom loss function
            custom_objects = {'euclidean_distance_loss': euclidean_distance_loss}
            return keras_load_model(model_path, custom_objects=custom_objects) 
        else:
            return keras_load_model(model_path)
    
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

    return numerical_cols, categorical_cols, positions

# Example usage: evaluate_model("NN_embedding_model_11") 
def evaluate_model(model_name):
    # Load varibles
    numerical_cols, categorical_cols, positions = extract_variables(model_name)

    # Load model
    model = load_model(f"models/{model_name}.h5", euclidean_distance_loss=True)

    # Prepare the input data and the DataFrame itself
    X_test_input, y_test = prepare_EL_input_data(test_frames_dfs, numerical_cols, categorical_cols, positions)
    test_prepared_frames_dfs = [prepare_df(test_frames_df, positions=positions) for test_frames_df in test_frames_dfs]

    # Concatenate the frames DataFrames into a single large DataFrame
    frames_concatenated_df = pd.concat(test_prepared_frames_dfs, ignore_index=True)

    # Make predictions using the loaded tf model
    predictions = model.predict(X_test_input)

    # Extract the predicted values
    x_future_pred = predictions[:, 0]
    y_future_pred = predictions[:, 1]

    # Add the predicted values to 'frames_concatenated_df'
    frames_concatenated_df['x_future_pred'] = x_future_pred
    frames_concatenated_df['y_future_pred'] = y_future_pred

    # Clip values to stay on the pitch
    frames_concatenated_df['x_future_pred'] = frames_concatenated_df['x_future_pred'].clip(lower=0, upper=pitch_length)
    frames_concatenated_df['y_future_pred'] = frames_concatenated_df['y_future_pred'].clip(lower=0, upper=pitch_width)

    # Smooth the predicted coordinates
    # smooth_predictions_xy(frames_df, alpha=0.95)

    error = total_error_loss(frames_concatenated_df)

    return error

# model_names = ["NN_model_v2"]
# for model_name in model_names:
#     error = evaluate_model(model_name)
#     print(f"{model_name}: {error}")

# Generate the corresponding txt filename
output_txt_filename = "models/26-03-24-model-testing.txt"

# Write the output directly to the txt file
with open(output_txt_filename, 'w') as f:
    for model_name in range(2, 12):
        model_name = f"NN_model_v{1}"
        error = evaluate_model(model_name)
        f.write(f"{model_name}: {error}\n")


# ### Test for different alpha values

# In[ ]:


# Example usage: run_model("NN_embedding_model_11") 
def run_model(model_name):
    # Load varibles
    numerical_cols, categorical_cols, positions = extract_variables(model_name)

    # Load model
    model = load_model(f"models/{model_name}.h5", euclidean_distance_loss=True)

    # Prepare the input data and the DataFrame itself
    X_test_input, y_test = prepare_EL_input_data(test_frames_dfs, numerical_cols, categorical_cols, positions)
    test_prepared_frames_dfs = [prepare_df(test_frames_df, positions=positions) for test_frames_df in test_frames_dfs]

    # Concatenate the frames DataFrames into a single large DataFrame
    frames_concatenated_df = pd.concat(test_prepared_frames_dfs, ignore_index=True)

    # Make predictions using the loaded tf model
    predictions = model.predict(X_test_input)

    # Extract the predicted values
    x_future_pred = predictions[:, 0]
    y_future_pred = predictions[:, 1]

    # Add the predicted values to 'frames_concatenated_df'
    frames_concatenated_df['x_future_pred'] = x_future_pred
    frames_concatenated_df['y_future_pred'] = y_future_pred

    # Clip values to stay on the pitch
    frames_concatenated_df['x_future_pred'] = frames_concatenated_df['x_future_pred'].clip(lower=0, upper=pitch_length)
    frames_concatenated_df['y_future_pred'] = frames_concatenated_df['y_future_pred'].clip(lower=0, upper=pitch_width)

    return frames_concatenated_df

# frames_df = run_model("NN_model_v2")
    
# alpha = 1.0
# for i in range(15):
#     frames_df = frames_df.copy()
#     smooth_predictions_xy(frames_df, alpha=alpha)
#     error = total_error_loss(frames_df)
#     print(f"{round(alpha, 2)}: {error}")
#     alpha -= 0.01


# In[ ]:


# Calculates how the average 'pred_error' varies with each value in 'column_to_analyze'
def find_column_variance(frames_df, column_to_analyze):
    # Convert 'pred_error' to numeric, coercing non-numeric values to NaN
    frames_df['pred_error'] = pd.to_numeric(frames_df['pred_error'], errors='coerce')

    # Group by 'column_to_analyze' and calculate the average 'pred_error'
    column_variance_df = frames_df.groupby(column_to_analyze)['pred_error'].mean().reset_index()

    # Round to 2 decimal places
    column_variance_df['pred_error'] = round(column_variance_df['pred_error'], 2)

    # Sort by 'column_to_analyze' in ascending order
    column_variance_df = column_variance_df.sort_values(by=column_to_analyze, ascending=True)

    # Return DataFrame with results
    return column_variance_df

# alpha = 1
# frames_df = run_model("NN_model_v2")
# total_error_loss(frames_df)
# column_to_analyze = 'position'

# # Call the function
# column_variance_df = find_column_variance(frames_df, column_to_analyze)

# # Print the DataFrame with results
# print(f"Average Pred Error per {column_to_analyze}:")
# print(column_variance_df)


# ## Predictive model 2
# ### LSTM model

# In[ ]:


def define_LSTM_model(input_shape):
    # Define the lengt of the sequence
    timesteps = 5 * FPS

    # Define the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(timesteps, input_shape[1])),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)  # Output layer with 2 units for x_future and y_future
    ])

    return model

def train_LSTM_model(X_train, y_train, X_val, y_val, val_frames_dfs):
    # Define the model
    model = define_LSTM_model(X_train.shape[1:])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=batch_size, verbose=0)

    # Save the trained model to disk
    model.save(get_next_model_filename("LSTM_model"))

    # Print the error using total_error_loss function
    train_error = total_error_loss(train_frames_dfs, include_ball=False, ball_has_to_be_in_motion=True)
    val_error = total_error_loss(val_frames_dfs, include_ball=False, ball_has_to_be_in_motion=True)
    print("Training Error:", train_error)
    print("Validation Error:", val_error)

def train_LSTM(train_frames_dfs, val_frames_dfs):
    # Prepare the data for training
    X_train, y_train = prepare_data(train_frames_dfs)

    # Prepare the data for validation
    X_val, y_val = prepare_data(val_frames_dfs)

    # Train the model
    train_model(X_train, y_train, X_val, y_val, val_frames_dfs)


# ### Visualize training results

# In[ ]:


# # Visualize training results
# model_name = 'NN_embedding_model_3'
# training_results = {
#     'loss': [2.0478146076202393, 2.0088889598846436, 2.0007753372192383, 1.9968146085739136, 1.9937269687652588, 1.9921172857284546, 1.990675687789917, 1.9893001317977905, 1.9881930351257324, 1.9875684976577759, 1.9872304201126099, 1.9865171909332275, 1.9859004020690918, 1.985435128211975, 1.9848004579544067, 1.983401894569397, 1.9824390411376953, 1.9820188283920288, 1.981824517250061, 1.9817743301391602],
#     'val_loss': [4.535243034362793, 4.51762580871582, 4.469428539276123, 4.436275482177734, 4.456634521484375, 4.815524578094482, 4.3103556632995605, 4.498797416687012, 4.790141582489014, 4.464589595794678, 4.674554347991943, 4.561259746551514, 4.533383369445801, 4.472135066986084, 4.466953754425049, 4.478504180908203, 4.723540782928467, 4.859069347381592, 4.496937274932861, 4.377903461456299]
# }

# visualize_training_results(training_results, model_name)

