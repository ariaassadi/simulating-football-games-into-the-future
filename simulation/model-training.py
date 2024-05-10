#!/usr/bin/env python
# coding: utf-8

# # Notebook for training predictive models
# ### Import packages

# In[ ]:


from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, BatchNormalization, Dropout, Reshape, LSTM
from tensorflow.keras.models import Model, load_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import date

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import time
import glob
import os

from utils import load_processed_frames, split_match_ids, embedding_config, get_next_model_filename, euclidean_distance_loss, total_error_loss, define_regularizers, prepare_EL_input_data, prepare_LSTM_input_data, create_embeddings, smooth_predictions_xy, run_model, evaluate_model, print_column_variance
from utils import prepare_df, add_can_be_sequentialized, extract_variables, load_tf_model, prepare_LSTM_df
from visualize_game import visualize_training_results
from settings import *


# ### Global variables

# In[ ]:


# Define parameters for model training
batch_size = 32
sequence_length = 10        # Sequence length for LSTM model
downsampling_factor = 5     # Keep every n:th frame


# ## Predictive model 1
# ### NN with Embedding layers
# Player-based model

# In[ ]:


# Define the architecture of the neural network model with embeddings layers
def define_NN_model(numerical_input_shape, categorical_cols, l1=0, l2=0):
    # Inputs for each categorical feature
    categorical_inputs = []
    categorical_flats = []
    for col in categorical_cols:
        # Replace spaces with underscores in the input name
        input_name = f'input_{col.replace(" ", "_")}'
        embedding_name = f'embedding_{col.replace(" ", "_")}'

        cat_input = Input(shape=(1,), name=input_name)  # Input for each categorical feature
        emb_layer = Embedding(
            input_dim=embedding_config[col]['n_categories'],
            output_dim=embedding_config[col]['output_dim'],
            input_length=1,
            name=embedding_name
        )(cat_input)
        flat_layer = Flatten()(emb_layer)
        categorical_inputs.append(cat_input)
        categorical_flats.append(flat_layer)

    # Prepare input layer for numerical data
    numerical_input = Input(shape=(numerical_input_shape,), name='numerical_input')

    # Concatenate all flattened embeddings with the numerical input
    concatenated_features = Concatenate()([*categorical_flats, numerical_input]) if categorical_flats else numerical_input

    # Dense layers
    regularizer = define_regularizers(l1, l2)  # Set regularizer
    dense_layer_1 = Dense(64, activation='relu', kernel_regularizer=regularizer)(concatenated_features)
    dense_layer_2 = Dense(32, activation='relu', kernel_regularizer=regularizer)(dense_layer_1)
    output_layer = Dense(len(y_cols), name='output_layer')(dense_layer_2)  # Output layer 'x_future' and 'y_future'

    # Building the model
    model = Model(inputs=[*categorical_inputs, numerical_input], outputs=output_layer)

    return model

def train_NN_model_with_embedding(train_ids, val_ids, numerical_cols, categorical_cols, positions, l1=0, l2=0, special_text=None):
    # Start time to later display how many seconds the execution too
    start_time = time.time()

    # Prepare inputs
    X_train_input, y_train = prepare_EL_input_data(train_ids, numerical_cols, categorical_cols, positions, downsampling_factor)
    X_val_input, y_val = prepare_EL_input_data(val_ids, numerical_cols, categorical_cols, positions, downsampling_factor)

    # Define the model
    model = define_NN_model(len(numerical_cols), categorical_cols, l1, l2)

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

        # Write the execution time
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        f.write(f"\nExecution time: {execution_time_minutes:.0f} minutes\n")

        # Write the training results
        f.write("\nTraining results:\n")
        for key, value in history.history.items():
            rounded_values = [round(v, 2) for v in value]
            f.write(f"{key}: {rounded_values}\n")


# In[ ]:


# Train the NN model with embedding layers
n_epochs = 1
n_matches = 10
categorical_cols = ['position']
positions=['Attacking Midfielder', 'Central Midfielder', 'Centre-Back', 'Defensive Midfielder', 'Forward', 'Full-Back', 'Goalkeeper', 'Wide Midfielder', 'Winger']
# positions = ['Goalkeeper', 'Centre-Back', 'Full-Back']
numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'angle_to_ball', 'tiredness', 'v_x_avg', 'v_y_avg']

train_ids, _, val_ids = split_match_ids(n_matches)
# train_NN_model_with_embedding(train_ids, val_ids, numerical_cols, categorical_cols, positions)


# ## Predictive model 2
# ### LSTM model
# Player-based model

# In[ ]:


# Define the architecture of the LSTM model with embeddings layers
def define_LSTM_model(numerical_input_shape, categorical_cols, sequence_length, l1=0, l2=0):
    categorical_inputs = []
    categorical_flats = []
    
    # Create inputs for each categorical feature
    for col in categorical_cols:
        input_name = f'input_{col.replace(" ", "_")}'
        embedding_name = f'embedding_{col.replace(" ", "_")}'

        cat_input = Input(shape=(1,), name=input_name)
        emb_layer = Embedding(
            input_dim=embedding_config[col]['n_categories'],
            output_dim=embedding_config[col]['output_dim'],
            input_length=1,
            name=embedding_name
        )(cat_input)
        flat_layer = Flatten()(emb_layer)
        categorical_inputs.append(cat_input)
        categorical_flats.append(flat_layer)

    # Prepare input layer for sequential numerical data
    numerical_input = Input(shape=(sequence_length, numerical_input_shape), name='numerical_input')
    lstm_layer = LSTM(64, return_sequences=False, name='lstm_numerical')(numerical_input)

    # Concatenate embeddings with numerical input
    if categorical_flats:
        concatenated_features = Concatenate()([*categorical_flats, lstm_layer])
    else:
        concatenated_features = lstm_layer  # Only use LSTM output if no categorical data

    # Dense layers
    regularizer = define_regularizers(l1, l2)  # Set regularizer
    dense_layer_1 = Dense(64, activation='relu', kernel_regularizer=regularizer)(concatenated_features)
    dense_layer_2 = Dense(32, activation='relu', kernel_regularizer=regularizer)(dense_layer_1)
    output_layer = Dense(len(y_cols), name='output_layer')(dense_layer_2)  # Output layer

    # Building the model
    model = Model(inputs=[*categorical_inputs, numerical_input], outputs=output_layer)

    return model

def train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions, l1=0, l2=0, special_text=None):
    # Prepare inputs
    start_time = time.time()
    X_train_input, y_train = prepare_LSTM_input_data(train_ids, numerical_cols, categorical_cols, sequence_length, positions, downsampling_factor)
    X_val_input, y_val = prepare_LSTM_input_data(val_ids, numerical_cols, categorical_cols, sequence_length, positions, downsampling_factor)
    data_preparation_time = time.time() - start_time
    print(f"Data preparation time: {data_preparation_time:.2f} seconds")

    # Define the model
    model = define_LSTM_model(len(numerical_cols), categorical_cols, sequence_length, l1, l2)

    # Compile the model
    model.compile(optimizer='adam', loss=euclidean_distance_loss)

    # Train the model with the corrected input format
    start_time = time.time()
    history = model.fit(X_train_input, y_train, validation_data=(X_val_input, y_val), epochs=n_epochs, batch_size=batch_size, verbose=2)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Save the trained model to disk
    model_filename = get_next_model_filename("LSTM_model")
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
        f.write(f"sequence_length={sequence_length}\n")
        f.write(f"numerical_cols={numerical_cols}\n")
        f.write(f"categorical_cols={categorical_cols}\n")
        f.write(f"positions={positions}\n")
        if l1 != 0: f.write(f"l1={l1}\n")
        if l2 != 0: f.write(f"l2={l2}\n")
        if special_text: f.write(f"{special_text}\n")

        # Write the execution time
        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        f.write(f"\nExecution time: {execution_time_minutes:.0f} minutes\n")

        # Write the training results
        f.write("\nTraining results:\n")
        for key, value in history.history.items():
            rounded_values = [round(v, 2) for v in value]
            f.write(f"{key}: {rounded_values}\n")


# In[ ]:


# tf.keras.backend.clear_session()
n_epochs = 5
n_matches = 60
positions=['Attacking Midfielder', 'Central Midfielder', 'Centre-Back', 'Defensive Midfielder', 'Forward', 'Full-Back', 'Goalkeeper', 'Wide Midfielder', 'Winger']
# positions=['Goalkeeper']
categorical_cols = ['position']
sequence_length = 10
numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'v_x_avg', 'v_y_avg']

# Only use 3 matches
# n_matches = 3
# train_ids, test_ids, val_ids = ['3acbc760-5bad-426c-ab7c-ef2d80f5ecf9'], ['b37b8919-e5bf-460e-a431-48feb878729f'], ['c820079c-2c34-485e-8b0a-73226152d986']

train_ids, _, val_ids = split_match_ids(n_matches)
train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-4)


# In[ ]:


numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'tiredness_short', 'v_x_avg', 'v_y_avg']
train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-5)
numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'v_x_avg', 'v_y_avg', 'distance_to_onside']
train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-5)
numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'angle_to_ball', 'tiredness', 'tiredness_short', 'v_x_avg', 'v_y_avg']
train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-5)
numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'angle_to_ball', 'tiredness', 'tiredness_short', 'v_x_avg', 'v_y_avg']
train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-5)


# In[ ]:


# n_matches = 160
# train_ids, _, val_ids = split_match_ids(n_matches)
# train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-6)

# n_matches = 280
# train_ids, _, val_ids = split_match_ids(n_matches)
# train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-6)

# n_matches = 560
# train_ids, _, val_ids = split_match_ids(n_matches)
# train_LSTM_model(train_ids, val_ids, numerical_cols, categorical_cols, sequence_length, positions=positions, l2=1e-6)


# ### Visualize training results

# In[ ]:


# # Visualize training results
# model_name = 'NN_model_v3'
# training_results = {
#     'loss': [2.0478146076202393, 2.0088889598846436, 2.0007753372192383, 1.9968146085739136, 1.9937269687652588, 1.9921172857284546, 1.990675687789917, 1.9893001317977905, 1.9881930351257324, 1.9875684976577759, 1.9872304201126099, 1.9865171909332275, 1.9859004020690918, 1.985435128211975, 1.9848004579544067, 1.983401894569397, 1.9824390411376953, 1.9820188283920288, 1.981824517250061, 1.9817743301391602],
#     'val_loss': [4.535243034362793, 4.51762580871582, 4.469428539276123, 4.436275482177734, 4.456634521484375, 4.815524578094482, 4.3103556632995605, 4.498797416687012, 4.790141582489014, 4.464589595794678, 4.674554347991943, 4.561259746551514, 4.533383369445801, 4.472135066986084, 4.466953754425049, 4.478504180908203, 4.723540782928467, 4.859069347381592, 4.496937274932861, 4.377903461456299]
# }

# visualize_training_results(training_results, model_name)

