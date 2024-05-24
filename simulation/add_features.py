"""
    File with functions for adding features to frames_df
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from scipy.signal import savgol_filter
from sklearn.pipeline import Pipeline

from pathlib import Path
import pandas as pd
import numpy as np
import random
import glob
import os

from visualize_game import visualize_game_animation, visualize_prediction_animation
from utils import google_sheet_to_df, load_processed_frames
from settings import *

"""
    Functions for adding features:
    - add_xy_future()
    - add_velocity_xy()
    - add_acceleration_xy()
    - add_average_velocity()
    - add_orientation()
    - add_ball_in_motion()
    - add_distance_to_ball()
    - add_angle_to_ball()
    - add_offside()
    - add_distance_to_onside()
    - add_FM_data()
    - add_tiredness()
    - add_tiredness_short_term()
"""

# Helper functions

# Add a vector with the 'x' coordinate of the ball
def add_x_ball(frames_df):
    # Create an 'x_ball' column with dtype float
    x_ball = pd.Series(dtype=float, index=frames_df['frame'])

    # Fill the values in 'x_ball' with the 'x' of the ball
    ball_positions = frames_df.loc[frames_df['team'] == 'ball', ['frame', 'x']].set_index('frame')['x']
    x_ball.update(ball_positions)

    # Add the 'x_ball' column to the DataFrame
    frames_df['x_ball'] = x_ball.values

# Add a vector with the shifted 'x' coordinate of the ball
def add_x_ball_prev(frames_df, frames_to_shift=1):
    # Create an 'x_ball_prev' column with dtype float
    x_ball = pd.Series(dtype=float, index=frames_df['frame'])
    
    # Shift the coordinates one frame
    ball_positions = frames_df.loc[frames_df['team'] == 'ball', ['frame', 'x']].set_index('frame')['x'].shift(frames_to_shift)
    x_ball.update(ball_positions)

    # Add the 'x_ball_prev' column to the DataFrame
    frames_df['x_ball_prev'] = x_ball.values

# Add a vector with the 'y' coordinate of the ball
def add_y_ball(frames_df):
    # Create a 'y_ball' column with dtype float
    y_ball = pd.Series(dtype=float, index=frames_df['frame'])

    # Fill the values in 'y_ball' with the 'y' of the ball
    ball_positions = frames_df.loc[frames_df['team'] == 'ball', ['frame', 'y']].set_index('frame')['y']
    y_ball.update(ball_positions)

    # Add the 'y_ball' column to the DataFrame
    frames_df['y_ball'] = y_ball.values

# Add a vector with the shifted 'y' coordinate of the ball
def add_y_ball_prev(frames_df, frames_to_shift=1):
    # Create a 'y_ball_prev' column with dtype float
    y_ball = pd.Series(dtype=float, index=frames_df['frame'])
    
    # Shift the coordinates one frame
    ball_positions = frames_df.loc[frames_df['team'] == 'ball', ['frame', 'y']].set_index('frame')['y'].shift(frames_to_shift)
    y_ball.update(ball_positions)

    # Add the 'y_ball_prev' column to the DataFrame
    frames_df['y_ball_prev'] = y_ball.values

# Here we go!

# Add the features x_future and y_future (the x and y coordinate of each player n frames into the future)
def add_xy_future(frames_df, n=50):
    # Shift the DataFrame by n frames for each player
    future_df = frames_df.groupby(['team', 'jersey_number', 'period']).shift(-n)

    # Merge the original DataFrame with the shifted DataFrame to get future coordinates
    frames_df[[f"x_future_{n}", f"y_future_{n}"]] = future_df[['x', 'y']]

    # Add the standard columns 'x_future' and 'y_future' if n corresponds to the numbers specified in the global variables file
    if n == FPS * seconds_into_the_future:
        frames_df[['x_future', 'y_future']] = future_df[['x', 'y']]

    return frames_df

# Add the features v_x and v_y (current velocity (m/s) in the x and y axis respectivly). delta_frames determines the time stamp
def add_velocity_xy(frames_df, delta_frames=1, smooth=False):
    # Initialzie empty columns
    frames_df['v_x'] = pd.Series(dtype='float64')
    frames_df['v_y'] = pd.Series(dtype='float64')

    # Create past_df by shifting frames_df by delta_frames for each player
    past_df = frames_df.groupby(['team', 'jersey_number', 'period']).shift(delta_frames)

    # Use the past coordinates to calculate the current velocity
    consecutive_frames = frames_df['frame'] == past_df['frame'] + delta_frames
    frames_df.loc[consecutive_frames, 'v_x'] = (frames_df['x'] - past_df['x']) * FPS / delta_frames
    frames_df.loc[consecutive_frames, 'v_y'] = (frames_df['y'] - past_df['y']) * FPS / delta_frames

    # Set velocity values to None if they exceed Usain Bolt's max speed
    usain_bolt_max_speed = 12.42
    frames_df['v_x'] = np.where(np.abs(frames_df['v_x']) > usain_bolt_max_speed, None, frames_df['v_x'])
    frames_df['v_y'] = np.where(np.abs(frames_df['v_y']) > usain_bolt_max_speed, None, frames_df['v_y'])

    # Smooth the velocities, if specified
    if smooth:
        # Apply Exponential Moving Average smoothing on the velocity columns
        def smooth_velocity_xy_ema(frames_df, alpha=0.93):
            # Group by unique combinations of 'team' and 'jersey_number'
            grouped = frames_df.groupby(['team', 'jersey_number', 'period'])
            
            # Apply the Exponential Moving Average filter to smooth the velocity
            def apply_ema(x):
                return x.ewm(alpha=alpha, adjust=False).mean()

            frames_df['v_x'] = grouped['v_x'].transform(apply_ema)
            frames_df['v_y'] = grouped['v_y'].transform(apply_ema)
            
        smooth_velocity_xy_ema(frames_df, alpha=0.93)

    # Round the results to two decimals
    frames_df['v_x'] = frames_df['v_x'].round(2)
    frames_df['v_y'] = frames_df['v_y'].round(2)

    return frames_df

# Add the features a_x and a_y (current velocity (m/s²) in the x and y axis respectivly). delta_frames determines the time stamp
def add_acceleration_xy(frames_df, delta_frames=1, smooth=False):
    # Initialzie empty columns
    frames_df['a_x'] = pd.Series(dtype='float64')
    frames_df['a_y'] = pd.Series(dtype='float64')

    # Create past_df by shifting frames_df by delta_frames for each player
    past_df = frames_df.groupby(['team', 'jersey_number', 'period']).shift(delta_frames)
    more_past_df = frames_df.groupby(['team', 'jersey_number', 'period']).shift(2 * delta_frames)

    # Use the past coordinates to calculate the current acceleration
    consecutive_frames = frames_df['frame'] == more_past_df['frame'] + 2 * delta_frames
    frames_df.loc[consecutive_frames, 'a_x'] = (frames_df['x'] - 2 * past_df['x'] + more_past_df['x']) * FPS / (delta_frames ** 2)
    frames_df.loc[consecutive_frames, 'a_y'] = (frames_df['y'] - 2 * past_df['y'] + more_past_df['y']) * FPS / (delta_frames ** 2)

    # Set acceleration values to None if they exceed Usain Bolt's max acceleration
    usain_bolt_max_acc = 9.5
    frames_df['a_x'] = np.where(np.abs(frames_df['a_x']) > usain_bolt_max_acc, None, frames_df['a_x'])
    frames_df['a_y'] = np.where(np.abs(frames_df['a_y']) > usain_bolt_max_acc, None, frames_df['a_y'])

    # Smooth the accelerations, if specified
    if smooth:
        # Apply Exponential Moving Average smoothing on the acceleration columns
        def smooth_acceleration_xy_ema(frames_df, alpha=0.2):
            # Group by unique combinations of 'team' and 'jersey_number'
            grouped = frames_df.groupby(['team', 'jersey_number', 'period'])
            
            # Apply the Exponential Moving Average filter to smooth the acceleration
            def apply_ema(x):
                return x.ewm(alpha=alpha, adjust=False).mean()

            frames_df['a_x'] = grouped['a_x'].transform(apply_ema)
            frames_df['a_y'] = grouped['a_y'].transform(apply_ema)
            
        smooth_acceleration_xy_ema(frames_df, alpha=0.2)

    # Round the results to two decimals
    frames_df['a_x'] = frames_df['a_x'].round(2)
    frames_df['a_y'] = frames_df['a_y'].round(2)

    return frames_df

# Add the vectors 'v_x_avg' and 'v_y_avg' with the average 'v_x' and 'v_y' of all players
def add_average_velocity(frames_df):
    # Drop 'v_x_avg' and 'v_y_avg' if they already exist
    frames_df = frames_df.drop(columns=['v_x_avg', 'v_y_avg'], errors='ignore')

    # Calculate the average velocity in each frame
    average_velocity = frames_df[frames_df['team'] != 'ball'].groupby('frame')[['v_x', 'v_y']].mean()

    # Round the average velocity to two decimal places
    average_velocity = average_velocity.round(2)

    # Add the results to frames_df
    frames_df = frames_df.merge(average_velocity, on='frame', suffixes=('', '_avg'))

    return frames_df

# Add the feature 'oreientation', using the veolicties to return the objects orientation (from 0 to 360 degrees)
# Oreintation 0 indicates that the object faces the goal to the right
# Orientation 180 indicates that the object faces the goal to the left
def add_orientation(frames_df):
    # Calculate orientation using arctan2 function
    frames_df['orientation'] = np.arctan2(frames_df['v_y'], frames_df['v_x']) * (180 / np.pi)
    
    # Convert orientation values to be in the range [0, 360)
    frames_df['orientation'] = (frames_df['orientation'] + 360) % 360

    return frames_df

# Add a vector indicating if the ball is in motion
def add_ball_in_motion(frames_df):
    # Initialize the 'ball_in_motion' column with False for all rows
    frames_df['ball_in_motion'] = False
    
    # Add 'x_ball' and 'y_ball' columns
    add_x_ball(frames_df)
    add_y_ball(frames_df)

    # Add 'x_ball_prev' and 'y_ball_prev' columns
    add_x_ball_prev(frames_df)
    add_y_ball_prev(frames_df)

    # Update the 'ball_in_motion' column to True if 'x_ball' or 'y_ball' exists, and any of the coordinates have changed
    frames_df.loc[(frames_df['x_ball'].notna()) & (frames_df['x_ball'] != frames_df['x_ball_prev']), 'ball_in_motion'] = True
    frames_df.loc[(frames_df['y_ball'].notna()) & (frames_df['y_ball'] != frames_df['y_ball_prev']), 'ball_in_motion'] = True

    # Drop unnecessary columns
    frames_df.drop(columns=['x_ball', 'x_ball_prev', 'y_ball', 'y_ball_prev'], inplace=True)

    return frames_df

# Add a vector with the distance to the ball
def add_distance_to_ball(frames_df):
    # Add 'x_ball' and 'y_ball' columns
    add_x_ball(frames_df)
    add_y_ball(frames_df)

    # Calculate the Euclidean distance from 'x' to 'x_ball' and 'y' to 'y_ball'
    frames_df['distance_to_ball'] = round(np.sqrt((frames_df['x'] - frames_df['x_ball'])**2 + (frames_df['y'] - frames_df['y_ball'])**2), 2)

    # Drop unnecessary columns
    frames_df.drop(columns=['x_ball', 'y_ball'], inplace=True)

    return frames_df

# Add a vector with the angle to the ball
def add_angle_to_ball(frames_df):
    # Add 'x_ball' and 'y_ball' columns
    add_x_ball(frames_df)
    add_y_ball(frames_df)

    # Calculate angle to the ball using arctan2 function
    frames_df['angle_to_ball'] = np.arctan2(frames_df['y_ball'] - frames_df['y'], frames_df['x_ball'] - frames_df['x']) * (180 / np.pi)
    
    # Convert angle to the ball values to be in the range [0, 360)
    frames_df['angle_to_ball'] = (frames_df['angle_to_ball'] + 360) % 360

    # Drop unnecessary columns
    frames_df.drop(columns=['x_ball', 'y_ball'], inplace=True)

    return frames_df

# Add a vector with the 'x' position of the second to last defender, for both team directions
def add_second_to_last_defender(frames_df):
    # Sort the DataFrame based on 'team', 'frame', 'x'
    sorted_frames_df = frames_df.sort_values(by=['team', 'frame', 'x']).copy()

    # Find the x coordinates of players attacking left and right for each frame
    x_players_attacking_left = sorted_frames_df[sorted_frames_df['team_direction'] == 'left'].groupby('frame')['x'].apply(list)
    x_players_attacking_right = sorted_frames_df[sorted_frames_df['team_direction'] == 'right'].groupby('frame')['x'].apply(list)

    # Find the x of the second to last defender
    x_second_to_last_player_left = x_players_attacking_left.apply(lambda x: x[-2] if len(x) >= 2 else pitch_length / 2)
    x_second_to_last_player_right = x_players_attacking_right.apply(lambda x: x[1] if len(x) >= 2 else pitch_length / 2)

    # Add 'x_second_to_last_player_left' and 'x_second_to_last_player_right' columns
    frames_df['x_second_to_last_player_left'] = x_second_to_last_player_left.reindex(frames_df['frame']).values
    frames_df['x_second_to_last_player_right'] = x_second_to_last_player_right.reindex(frames_df['frame']).values

# Add a vector with the 'offside_line'
def add_offside_line(frames_df):
    # Create a vector for the values of the half way line
    frames_df['half_way_line'] = pitch_length / 2

    # Add 'x_ball' column and fill None values with the half way line
    add_x_ball(frames_df)
    frames_df['x_ball'].fillna(pitch_length / 2, inplace=True)

    # Add 'x_second_to_last_player_left' and 'x_second_to_last_player_right' columns
    add_second_to_last_defender(frames_df)

    # Update 'offside_line' column based on team direction
    frames_df['offside_line'] = np.where(
        frames_df['team_direction'] == 'right',
        # If team_direction is 'right', the offside line will be the max value of the second to last defender, ball, and half way line
        np.maximum.reduce([frames_df['x_second_to_last_player_left'], frames_df['x_ball'], frames_df['half_way_line']]),
        # If team_direction is 'left', the offside line will be the min value of the second to last defender, ball, and half way line
        np.minimum.reduce([frames_df['x_second_to_last_player_right'], frames_df['x_ball'], frames_df['half_way_line']])
    )

    # Set offside line to half way line if the 'team' is ball
    frames_df.loc[frames_df['team'] == 'ball', 'offside_line'] = pitch_length / 2

    # Drop unnecessary columns
    frames_df.drop(columns=['half_way_line', 'x_ball', 'x_second_to_last_player_left', 'x_second_to_last_player_right'], inplace=True)

# Add a vector the sets the value to 'offside_line' if a player is standing in an offside position
def add_offside(frames_df):
    # Add 'offside_line' column
    add_offside_line(frames_df)
    
    # Create the empty column
    frames_df['offside'] = None
    
    # Fill the 'offside' column based on conditions
    frames_df.loc[(frames_df['team_direction'] == 'right') & (frames_df['x'] > frames_df['offside_line']), 'offside'] = frames_df['offside_line']
    frames_df.loc[(frames_df['team_direction'] == 'left') & (frames_df['x'] < frames_df['offside_line']), 'offside'] = frames_df['offside_line']

    # Drop the 'offside_line' column
    frames_df.drop(columns=['offside_line'], inplace=True)

    return frames_df

# Add a vector with the distance to the offside line
def add_distance_to_onside(frames_df):
    # Check if 'offside' column exists and add it if it doesn't
    if 'offside' not in frames_df.columns:
        frames_df = add_offside(frames_df)

    # Calculate the distance to the offside line
    frames_df['distance_to_onside'] = frames_df['x'] - frames_df['offside']

    # Fill NaN values in the 'distance_to_onside' column with 0
    frames_df['distance_to_onside'].fillna(0, inplace=True)

    return frames_df

# Add data from the Football Manager
def add_FM_data(frames_df, fm_players_df):
    # List of feature to add together with their corresponding data types
    fm_features = ['Nationality', 'Height', 'Weight', 'Acc', 'Pac', 'Sta', 'Position', 'Specific Position']
    fm_types = ['category', 'float64', 'Int8', 'Int8', 'Int8', 'Int8', 'category', 'category']
    
    # Create a dictionary to map features to types
    type_dict = dict(zip(fm_features, fm_types))

    # Merge Football Manager data with frames_df
    merged_df = frames_df.merge(fm_players_df[['Player', 'Team'] + fm_features],
                                left_on=['player', 'team_name'],
                                right_on=['Player', 'Team'],
                                how='left')

    # Add each feature as a vector in frames_df
    for feature in fm_features:
        # Keep column names consistent with lowercase in frames_df
        feature_name = feature.lower().replace(" ", "_")

        # Apply type conversion directly during the assignment
        frames_df[feature_name] = merged_df[feature].astype(type_dict[feature])

    return frames_df

# Add a vector indicating how tired the player is
def add_tiredness(frames_df):
    # Calculate tiredness using the formula above
    frames_df['tiredness'] = ((frames_df['distance_ran'] / 1000) + (frames_df['minute'] / 20) + frames_df['period'] - 1) * (1 - (frames_df['sta'] / 20))

    return frames_df

# Add a vector with the short-term tiredness of each player
def add_tiredness_short_term(frames_df, window=FPS*20):
    # Calculate the average absolute acceleration
    frames_df['a_abs'] = np.sqrt(frames_df['a_x'] ** 2 + frames_df['a_y'] ** 2)
    frames_df['a_abs_avg_20s'] = round(frames_df.groupby(['team', 'jersey_number', 'period'])['a_abs'].transform(lambda x: x.rolling(window=window, min_periods=1).mean()), 3)

    # Create past_df by shifting frames_df by window for each player
    past_df = frames_df.groupby(['team', 'jersey_number', 'period']).shift(window)
    
    # Calculate the distance ran over the time window
    consecutive_frames = frames_df['frame'] == past_df['frame'] + window
    frames_df.loc[consecutive_frames, 'distance_ran_time_window'] = frames_df['distance_ran'] - past_df['distance_ran']

    # Calculate the tiredness using the given formula and fill NaN values with 0
    frames_df['tiredness_short'] = round((frames_df['distance_ran_time_window'] / 100) + frames_df['a_abs_avg_20s'], 2)
    frames_df['tiredness_short'] = frames_df['tiredness_short'].fillna(0)

    # Set 'tiredness_short' to None for the ball
    frames_df.loc[frames_df['team'] == 'ball', 'tiredness_short'] = None
    
    # Drop the temporary columns
    frames_df.drop(columns=['a_abs', 'a_abs_avg_20s', 'distance_ran_time_window'], inplace=True)

    return frames_df
    