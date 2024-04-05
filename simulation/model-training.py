#!/usr/bin/env python
# coding: utf-8

# # Notebook for training predictive models
# ### Import packages

# In[1]:


from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, BatchNormalization, Dropout, Reshape, LSTM
from tensorflow.keras.models import Model, load_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import date

import tensorflow as tf
import pandas as pd
import numpy as np
import random
import glob
import os

from utils import load_processed_frames, split_match_ids, get_next_model_filename, euclidean_distance_loss, total_error_loss, define_regularizers, prepare_EL_input_data, prepare_LSTM_input_data, create_embeddings, smooth_predictions_xy, run_model, evaluate_model, print_column_variance
from utils import prepare_df, add_can_be_sequentialized
from visualize_game import visualize_training_results
from settings import *


# ### Global variables

# In[2]:


# Define numerical, categorical, and y columns
numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y']
categorical_cols = ['team_direction', 'role']

# Define parameters for model training
n_epochs = 1
batch_size = 32
n_matches = 40
sequence_length = 10    # Sequence length for LSTM model


# ### Load frames

# In[3]:


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


# ## Predictive model 1
# ### NN with Embedding layers
# Player-based model

# In[4]:


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

def train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, numerical_cols, categorical_cols, positions=[], l1=0, l2=0, special_text=None):
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


# In[5]:


# Train the NN model with embedding layers
n_epochs = 1
categorical_cols = ['position']
# positions = ["Attacking Midfielder", "Central Midfielder", "Centre-Back", "Defensive Midfielder", "Forward", "Full-Back", "Goalkeeper", "Wide Midfielder", "Winger"]
positions = ["Central Midfielder", "Winger"]

numerical_cols = ['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball']
# train_NN_model_with_embedding(train_frames_dfs, val_frames_dfs, numerical_cols, categorical_cols, positions=positions)


# ### Evaluate model

# In[6]:


# model_names = ["LSTM_model_v2"]
# for model_name in model_names:
#     error = evaluate_model(test_frames_dfs, model_name)
#     print(f"{model_name}: {error}")

# Test different alpha values

# frames_df = run_model(test_frames_dfs, "LSTM_model_v5")
# alpha = 1.0
# for i in range(10):
#     frames_df = frames_df.copy()
#     smooth_predictions_xy(frames_df, alpha=alpha)
#     error = total_error_loss(frames_df)
#     print(f"{round(alpha, 2)}: {error}")
#     alpha -= 0.01


# In[35]:


# # Print column variance for position
# column_to_analyze = 'position'
# model_name = "LSTM_model_v9"
# print_column_variance(val_frames_dfs, model_name, column_to_analyze)


# ## Predictive model 2
# ### LSTM model
# Player-based model

# In[ ]:


# Define the NN model with LSTM layer
def define_LSTM_model(numerical_input_shape, sequence_length, l1=0, l2=0):  
    # Handle case where we have categorical columns
    if categorical_cols:
        # Create embeddings for categorical data
        categorical_inputs, categorical_flats = create_embeddings(categorical_cols)
        
        # Input for numerical data
        numerical_input = Input(shape=(sequence_length, numerical_input_shape), name='numerical_input')

        # Processing sequence with LSTM
        lstm_out = LSTM(64)(numerical_input)

        # Assuming we want to concatenate LSTM output with categorical embeddings
        # Note: This might need adjustment based on how you want to use categorical data
        concatenated_features = Concatenate()([lstm_out] + categorical_flats)

        model_inputs = categorical_inputs + [numerical_input]

    # Handle case where we only have numerical columns3
    else:
        numerical_input = Input(shape=(sequence_length, numerical_input_shape), name='numerical_input')  # Numerical input
        lstm_layer = LSTM(64)(numerical_input)  # LSTM layer directly using numerical input
        concatenated_features = lstm_layer  # Directly use LSTM output
        model_inputs = [numerical_input]  # Model inputs

    # Dense layers
    regularizer = define_regularizers(l1, l2)  # Set regularizer
    dense_layer_1 = Dense(64, activation='relu', kernel_regularizer=regularizer)(concatenated_features)
    dense_layer_2 = Dense(32, activation='relu', kernel_regularizer=regularizer)(dense_layer_1)

    output_layer = Dense(2)(dense_layer_2)  # Output layer for x_future and y_future
    model = Model(inputs=model_inputs, outputs=output_layer)  # Build model

    return model

def train_LSTM_model(train_frames_dfs, val_frames_dfs, sequence_length, positions=[], l1=0, l2=0, special_text=None):
    # Prepare inputs
    X_train_input, y_train = prepare_LSTM_input_data(train_frames_dfs, numerical_cols, categorical_cols, sequence_length, positions=positions)
    X_val_input, y_val = prepare_LSTM_input_data(val_frames_dfs, numerical_cols, categorical_cols, sequence_length, positions=positions)

    # Define the model
    model = define_LSTM_model(numerical_input_shape=len(numerical_cols), sequence_length=sequence_length, l1=l1, l2=l2)

    # Compile the model
    model.compile(optimizer='adam', loss=euclidean_distance_loss)

    # Train the model with the corrected input format
    history = model.fit(X_train_input, y_train, validation_data=(X_val_input, y_val), epochs=n_epochs, batch_size=batch_size, verbose=2)

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

        # Write the training results
        f.write("\nTraining results:\n")
        for key, value in history.history.items():
            rounded_values = [round(v, 2) for v in value]
            f.write(f"{key}: {rounded_values}\n")


# In[34]:


n_epochs = 5
positions=['Attacking Midfielder', 'Central Midfielder', 'Centre-Back', 'Defensive Midfielder', 'Forward', 'Full-Back', 'Goalkeeper', 'Wide Midfielder', 'Winger']
# positions=['Goalkeeper', 'Winger']
categorical_cols = ["position"]
sequence_length = 10

numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball']
train_LSTM_model(train_frames_dfs, val_frames_dfs, sequence_length, positions=positions, l2=1e-6)

numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness']
train_LSTM_model(train_frames_dfs, val_frames_dfs, sequence_length, positions=positions, l2=1e-6)

numerical_cols=['x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'distance_to_ball', 'tiredness', 'height']
train_LSTM_model(train_frames_dfs, val_frames_dfs, sequence_length, positions=positions, l2=1e-6)


# In[ ]:


# model_name = "LSTM_model_v8"
# error = evaluate_model(test_frames_dfs, model_name)
# print(f"{model_name}: {error}")


# ### Visualize training results

# In[ ]:


# # Visualize training results
# model_name = 'NN_embedding_model_3'
# training_results = {
#     'loss': [2.0478146076202393, 2.0088889598846436, 2.0007753372192383, 1.9968146085739136, 1.9937269687652588, 1.9921172857284546, 1.990675687789917, 1.9893001317977905, 1.9881930351257324, 1.9875684976577759, 1.9872304201126099, 1.9865171909332275, 1.9859004020690918, 1.985435128211975, 1.9848004579544067, 1.983401894569397, 1.9824390411376953, 1.9820188283920288, 1.981824517250061, 1.9817743301391602],
#     'val_loss': [4.535243034362793, 4.51762580871582, 4.469428539276123, 4.436275482177734, 4.456634521484375, 4.815524578094482, 4.3103556632995605, 4.498797416687012, 4.790141582489014, 4.464589595794678, 4.674554347991943, 4.561259746551514, 4.533383369445801, 4.472135066986084, 4.466953754425049, 4.478504180908203, 4.723540782928467, 4.859069347381592, 4.496937274932861, 4.377903461456299]
# }

# visualize_training_results(training_results, model_name)

