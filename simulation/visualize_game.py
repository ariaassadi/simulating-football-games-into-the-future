from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
import imageio
import os

from settings import *

# Define variables for plotting
draw_velocities = True
draw_orientations = False
draw_angle_to_ball = False
draw_shirt_numbers = False
draw_offsides = False
draw_prediction_lines = True

player_size = 300
ball_size = 200
predicted_player_size = 0.8 * player_size
predicted_ball_size = 0.8 * ball_size
offside_line_thickness = 4

# Original colors
color_home_team = ajax_red
color_away_team = ajax_yellow
color_ball = ajax_black
color_edge = ajax_black

# Lighter shades
color_home_team_light = ajax_red_light
color_away_team_light = ajax_yellow_light
color_ball_light = ajax_black_light

# Helper functions
def get_frame_data(frames_df, frame):
    home_team_frame_df = frames_df[(frames_df['team'] == 'home_team') & (frames_df['frame'] == frame)].copy()
    away_team_frame_df = frames_df[(frames_df['team'] == 'away_team') & (frames_df['frame'] == frame)].copy()
    ball_frame_df = frames_df[(frames_df['team'] == 'ball') & (frames_df['frame'] == frame)].copy()
    return home_team_frame_df, away_team_frame_df, ball_frame_df

# Draw a legend underneth the pitch
def draw_legend(ax, home_team, away_team):
    y_legend = -2
    # Home team
    ax.scatter(pitch_length/2 -28, y_legend, s=player_size, color=color_home_team, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 - 26, y_legend, home_team, ha='left', va='center', fontsize=14)

    # Ball
    ax.scatter(pitch_length/2 - 2, y_legend, s=ball_size, color=color_ball, edgecolors=color_edge, linewidth=1.8, zorder=3)
    ax.text(pitch_length/2 - 0, y_legend, 'Ball', ha='left', va='center', fontsize=14)

    # Away team
    ax.scatter(pitch_length/2 + 14, y_legend, s=player_size, color=color_away_team, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 + 16, y_legend, away_team, ha='left', va='center', fontsize=14)

# Create a gif file with a game snippet
def visualize_game_animation(frames_df, start_frame, end_frame):
    # Find DataFrames for each team and the ball
    home_team_df = frames_df[frames_df['team'] == 'home_team'].copy()
    away_team_df = frames_df[frames_df['team'] == 'away_team'].copy()
    ball_df = frames_df[frames_df['team'] == 'ball'].copy()

    # Find the name of the home and away team
    home_team = home_team_df.iloc[0]['team_name']
    away_team = away_team_df.iloc[0]['team_name']

    # Plot pitch
    pitch = Pitch(pitch_type='uefa', goal_type='box', line_color=color_edge)
    fig, ax = pitch.grid(grid_height=0.96, title_height=0.02, axis=False, endnote_height=0.02, title_space=0, endnote_space=0)

    # Add title
    fig.suptitle(f"{home_team} - {away_team}", fontsize=18)

    # Define function to update scatter plots for a specific frame
    def update_scatter(frame):
        # Clear previous scatters, text, etc
        ax['pitch'].collections.clear()
        ax['pitch'].texts.clear()
        ax['pitch'].patches.clear()

        # Create DataFrames for home team, away team, and ball
        home_team_frame_df = home_team_df[home_team_df['frame'] == frame]
        away_team_frame_df = away_team_df[away_team_df['frame'] == frame]
        ball_frame_df = ball_df[ball_df['frame'] == frame]

        # Plot the time stamp
        time_stamp = f"{home_team_frame_df.iloc[0]['minute']}:{home_team_frame_df.iloc[0]['second']}:{home_team_frame_df.iloc[0]['frame'] % FPS}"
        ax['pitch'].text(0, 70, time_stamp, ha='left', fontsize=18)

        # Scatter the positions of the home players
        ax['pitch'].scatter(home_team_frame_df['x'], home_team_frame_df['y'], s=player_size, color=color_home_team, edgecolors=color_edge, linewidth=1.8, label=home_team, zorder=2)

        # Scatter the positions of the away players
        ax['pitch'].scatter(away_team_frame_df['x'], away_team_frame_df['y'], s=player_size, color=color_away_team, edgecolors=color_edge, linewidth=1.8, label=away_team, zorder=2)

        # Draw the ball
        ax['pitch'].scatter(ball_frame_df['x'], ball_frame_df['y'], s=ball_size, color=color_ball, edgecolors=color_edge, linewidth=1.8, label='Ball', zorder=3)

        # Draw jersey numbers inside the scatters
        if draw_shirt_numbers:
            for _, player in home_team_frame_df.iterrows():
                ax['pitch'].text(player['x'], player['y'], str(player['jersey_number']), fontsize=10, fontweight='semibold', ha='center', va='center', zorder=5)
            for _, player in away_team_frame_df.iterrows():
                ax['pitch'].text(player['x'], player['y'], str(player['jersey_number']), fontsize=10, fontweight='semibold', ha='center', va='center', zorder=5)
            
        # Draw arrows for the velocity of each player
        if draw_velocities:
            # Draw the velocity for home players with valid velocities
            for _, row in home_team_frame_df[~home_team_frame_df[['v_x', 'v_y']].isna().any(axis=1)].iterrows():
                ax['pitch'].arrow(row['x'], row['y'], row['v_x'], row['v_y'], color=color_edge, head_width=0.5, head_length=0.5, zorder=1)
        
            # Draw the velocity for away players with valid velocities
            for _, row in away_team_frame_df[~away_team_frame_df[['v_x', 'v_y']].isna().any(axis=1)].iterrows():
                ax['pitch'].arrow(row['x'], row['y'], row['v_x'], row['v_y'], color=color_edge, head_width=0.5, head_length=0.5, zorder=1)
        
        # Draw arrows for the angle_to_ball for each player
        if draw_angle_to_ball:
            arrow_length = 4

            # Draw the angle_to_ball for home players with valid velocities
            for _, row in home_team_frame_df[~home_team_frame_df['angle_to_ball'].isna()].iterrows():
                # Calculate the end point of the arrow based on orientation
                dx = np.cos(np.radians(row['angle_to_ball'])) * arrow_length
                dy = np.sin(np.radians(row['angle_to_ball'])) * arrow_length
                ax['pitch'].arrow(row['x'], row['y'], dx, dy, color=color_edge, head_width=0.5, head_length=0.5, zorder=1)
            
            # Draw the angle_to_ball for away players with valid velocities
            for _, row in away_team_frame_df[~away_team_frame_df['angle_to_ball'].isna()].iterrows():
                # Calculate the end point of the arrow based on orientation
                dx = np.cos(np.radians(row['angle_to_ball'])) * arrow_length
                dy = np.sin(np.radians(row['angle_to_ball'])) * arrow_length
                ax['pitch'].arrow(row['x'], row['y'], dx, dy, color=color_edge, head_width=0.5, head_length=0.5, zorder=1)
        
        # Draw arrows for the orientation of each player
        if draw_orientations:
            arrow_length = 4

            # Draw the orientation for home players with valid velocities
            for _, row in home_team_frame_df[~home_team_frame_df['orientation'].isna()].iterrows():
                # Calculate the end point of the arrow based on orientation
                dx = np.cos(np.radians(row['orientation'])) * arrow_length
                dy = np.sin(np.radians(row['orientation'])) * arrow_length
                ax['pitch'].arrow(row['x'], row['y'], dx, dy, color=color_edge, head_width=0.5, head_length=0.5, zorder=1)
            
            # Draw the orientation for away players with valid velocities
            for _, row in away_team_frame_df[~away_team_frame_df['orientation'].isna()].iterrows():
                # Calculate the end point of the arrow based on orientation
                dx = np.cos(np.radians(row['orientation'])) * arrow_length
                dy = np.sin(np.radians(row['orientation'])) * arrow_length
                ax['pitch'].arrow(row['x'], row['y'], dx, dy, color=color_edge, head_width=0.5, head_length=0.5, zorder=1)

        # Draw offside lines and mark players that are standing in offside positions 
        if draw_offsides:
            # Scatter the offside players again, but with other colors
            ax['pitch'].scatter(home_team_frame_df[home_team_frame_df["offside"].notna()]['x'], home_team_frame_df[home_team_frame_df["offside"].notna()]['y'], s=player_size, color=ajax_grey, alpha=0.35, edgecolors=color_edge, linewidth=1.8, zorder=4)
            ax['pitch'].scatter(away_team_frame_df[away_team_frame_df["offside"].notna()]['x'], away_team_frame_df[away_team_frame_df["offside"].notna()]['y'], s=player_size, color=ajax_grey, alpha=0.35, edgecolors=color_edge, linewidth=1.8, zorder=4)

            # Draw offisde line(s)
            x_home_team_offside = home_team_frame_df["offside"].max()
            if not pd.isna(x_home_team_offside):
                offside_line_home = patches.Rectangle((x_home_team_offside - 0.4, 0), width=0.8, height=pitch_width, color=ajax_yellow, zorder=0)
                ax['pitch'].add_patch(offside_line_home)

            x_away_team_offside = away_team_frame_df["offside"].max()
            if not pd.isna(x_away_team_offside):
                offside_line_away = patches.Rectangle((x_away_team_offside - 0.4, 0), width=0.8, height=pitch_width, color=ajax_yellow, zorder=0)
                ax['pitch'].add_patch(offside_line_away)

        # Add legend
        draw_legend(ax['pitch'], home_team, away_team)

    # Create the animation
    animation = FuncAnimation(fig, update_scatter, frames=range(start_frame, end_frame+1), repeat=False, interval=40)

    # Specify the GIF file path
    gif_name = f"animations/{home_team}_vs_{away_team}_frames_{start_frame}-{end_frame}.gif"

    # Save the animation as a GIF file
    animation.save(gif_name, writer='pillow')

# Example usage:
# visualize_game_snippet(frames_df, 0, 1600)

# Draw a legend underneth the pitch
def draw_prediction_legend(ax, model_name, home_team, away_team):
    y_legend = -1.35
    y_predicted = y_legend - 1.85

    # Home team
    ax.scatter(pitch_length/2 - 28, y_legend, s=player_size*0.5, color=color_home_team, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 - 26, y_legend, home_team, ha='left', va='center', fontsize=14)

    ax.scatter(pitch_length/2 - 28, y_predicted, s=predicted_player_size*0.5, color=color_home_team_light, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 - 26, y_predicted, 'Predicted', ha='left', va='center', fontsize=14)

    # Away team
    ax.scatter(pitch_length/2 + 16, y_legend, s=player_size*0.5, color=color_away_team, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 + 18, y_legend, away_team, ha='left', va='center', fontsize=14)

    ax.scatter(pitch_length/2 + 16, y_predicted, s=predicted_player_size*0.5, color=color_away_team_light, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 + 18, y_predicted, 'Predicted', ha='left', va='center', fontsize=14)

    # Ball
    ax.scatter(pitch_length/2 - 2, y_legend, s=ball_size*0.5, color=color_ball, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 - 0, y_legend, 'Ball', ha='left', va='center', fontsize=14)

    ax.scatter(pitch_length/2 - 2, y_predicted, s=predicted_ball_size*0.5, color=color_ball_light, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax.text(pitch_length/2 - 0, y_predicted, 'Predicted', ha='left', va='center', fontsize=14)

# Visualize the prediction made in a given frame
def visualize_frame_prediction(frames_df, frame, model_name):
    # Find DataFrames for each team and the ball
    home_team_frame_df = frames_df[(frames_df['team'] == 'home_team') & (frames_df['frame'] == frame)].copy()
    away_team_frame_df = frames_df[(frames_df['team'] == 'away_team') & (frames_df['frame'] == frame)].copy()
    ball_frame_df = frames_df[(frames_df['team'] == 'ball') & (frames_df['frame'] == frame)].copy()

    # Find the name of the home and away team
    home_team = home_team_frame_df.iloc[0]['team_name']
    away_team = away_team_frame_df.iloc[0]['team_name']

    # Plot pitch
    pitch = Pitch(pitch_type='uefa', goal_type='box', line_color=color_edge)
    fig, ax = pitch.grid(grid_height=0.95, title_height=0.025, axis=False, endnote_height=0.025, title_space=0, endnote_space=0)

    # Adjust the x- and y-axis limits
    ax['pitch'].set_ylim(0, pitch_length)
    ax['pitch'].set_ylim(-6, pitch_width + 3.4)

    # Add title
    fig.suptitle(f"{home_team} - {away_team}", fontsize=18)

    # Plot the time stamp
    time_stamp = f"{home_team_frame_df.iloc[0]['minute']}:{home_team_frame_df.iloc[0]['second']}:{home_team_frame_df.iloc[0]['frame'] % FPS}"
    ax['pitch'].text(0, 70, time_stamp, ha='left', fontsize=18)

    # Scatter the true positions
    ax['pitch'].scatter(home_team_frame_df['x_future'], home_team_frame_df['y_future'], s=player_size, color=color_home_team, edgecolors=color_edge, linewidth=1.8, zorder=3)
    ax['pitch'].scatter(away_team_frame_df['x_future'], away_team_frame_df['y_future'], s=player_size, color=color_away_team, edgecolors=color_edge, linewidth=1.8, zorder=3)
    ax['pitch'].scatter(ball_frame_df['x_future'], ball_frame_df['y_future'], s=ball_size, color=color_ball, edgecolors=color_edge, linewidth=1.8, zorder=6)

    # Draw jersey numbers inside the true predictions
    if draw_shirt_numbers:
        for _, player in home_team_frame_df.iterrows():
            ax['pitch'].text(player['x_future'], player['y_future'], str(player['jersey_number']), fontsize=10, fontweight='semibold', ha='center', va='center', zorder=4)
        for _, player in away_team_frame_df.iterrows():
            ax['pitch'].text(player['x_future'], player['y_future'], str(player['jersey_number']), fontsize=10, fontweight='semibold', ha='center', va='center', zorder=4)
        
    # Scatter the predicted positions with an opacity
    ax['pitch'].scatter(home_team_frame_df['x_future_pred'], home_team_frame_df['y_future_pred'], s=predicted_player_size, color=color_home_team_light, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax['pitch'].scatter(away_team_frame_df['x_future_pred'], away_team_frame_df['y_future_pred'], s=predicted_player_size, color=color_away_team_light, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ax['pitch'].scatter(ball_frame_df['x_future_pred'], ball_frame_df['y_future_pred'], s=predicted_ball_size, color=color_ball_light, edgecolors=color_edge, linewidth=1.8, zorder=5)

    # Plot lines between true positions and predicted positions with the 'pred_error' over each line
    if draw_prediction_lines:
        for index, row in home_team_frame_df.iterrows():
            ax['pitch'].plot([row['x_future'], row['x_future_pred']], [row['y_future'], row['y_future_pred']], color=color_edge, linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)

        for index, row in away_team_frame_df.iterrows():
            ax['pitch'].plot([row['x_future'], row['x_future_pred']], [row['y_future'], row['y_future_pred']], color=color_edge, linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)

        for index, row in ball_frame_df.iterrows():
            ax['pitch'].plot([row['x_future'], row['x_future_pred']], [row['y_future'], row['y_future_pred']], color=color_edge, linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)

    # Draw the average prediction error in the bottom left corner
    average_pred_error = frames_df[frames_df['frame'] == frame]['pred_error'].mean()
    if average_pred_error is not None:
        average_pred_error_txt = f"Avg. error: {round(average_pred_error, 2)} m"
    ax['pitch'].text(pitch_length - 15, -3.5, average_pred_error_txt, ha='left', va='center', fontsize=14)

    # Add legend
    draw_prediction_legend(ax['pitch'], model_name, home_team, away_team)

    # Save figure
    img_path = f"images/predictions/{home_team}_vs_{away_team}_frame_{frame}_{model_name}.png"
    fig.savefig(img_path)

# Example usage:
# visualize_frame_prediction(frames_df, 100, "naive_static")

def visualize_prediction_animation(frames_df, start_frame, end_frame, model_name):
    # Find DataFrames for each team and the ball
    home_team_df = frames_df[frames_df['team'] == 'home_team'].copy()
    away_team_df = frames_df[frames_df['team'] == 'away_team'].copy()
    ball_df = frames_df[frames_df['team'] == 'ball'].copy()

    # Find the name of the home and away team
    home_team = home_team_df.iloc[0]['team_name']
    away_team = away_team_df.iloc[0]['team_name']

    # Plot pitch
    pitch = Pitch(pitch_type='uefa', goal_type='box', line_color=color_edge)
    fig, ax = pitch.grid(grid_height=0.96, title_height=0.02, axis=False, endnote_height=0.02, title_space=0, endnote_space=0)

    # Add title
    fig.suptitle(f"{home_team} - {away_team}", fontsize=18)

    # Add model name
    ax['pitch'].text(pitch_length, 70, f"{model_name}", ha='right', va='center', fontsize=16)

    # Draw the legend
    draw_prediction_legend(ax['pitch'], model_name, home_team, away_team)

    # Decide how each scatter should look for true coordinates
    home_team_true_scatter = ax['pitch'].scatter([], [], s=player_size, color=color_home_team, edgecolors=color_edge, linewidth=1.8, zorder=3)
    away_team_true_scatter = ax['pitch'].scatter([], [], s=player_size, color=color_away_team, edgecolors=color_edge, linewidth=1.8, zorder=3)
    ball_true_scatter = ax['pitch'].scatter([], [], s=ball_size, color=color_ball, edgecolors=color_edge, linewidth=1.8, zorder=6)

    # Decide how each scatter should look for pedicted coordinates
    home_team_pred_scatter = ax['pitch'].scatter([], [], s=predicted_player_size, color=color_home_team_light, edgecolors=color_edge, linewidth=1.8, zorder=2)
    away_team_pred_scatter = ax['pitch'].scatter([], [], s=predicted_player_size, color=color_away_team_light, edgecolors=color_edge, linewidth=1.8, zorder=2)
    ball_pred_scatter = ax['pitch'].scatter([], [], s=predicted_ball_size, color=color_ball_light, edgecolors=color_edge, linewidth=1.8, zorder=5)

    # Decide how each frame-dependent text should look
    frame_text = ax['pitch'].text(0, 70, '', ha='left', va='center', fontsize=16)
    error_text = ax['pitch'].text(pitch_length, -2, '', ha='right', va='center', fontsize=14)

    # Define LineCollection objects for each team and the ball
    home_team_error_lines = LineCollection([], color=color_edge, linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
    away_team_error_lines = LineCollection([], color=color_edge, linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)
    ball_error_lines = LineCollection([], color=color_edge, linewidth=1.5, linestyle='--', alpha=0.7, zorder=1)

    # Add LineCollection objects to the plot
    ax['pitch'].add_collection(home_team_error_lines)
    ax['pitch'].add_collection(away_team_error_lines)
    ax['pitch'].add_collection(ball_error_lines)

    # Function for updating each frame
    def update(frame):
        home_team_frame_df, away_team_frame_df, ball_frame_df = get_frame_data(frames_df, frame)

        home_team_true_scatter.set_offsets(np.c_[home_team_frame_df['x_future'], home_team_frame_df['y_future']])
        away_team_true_scatter.set_offsets(np.c_[away_team_frame_df['x_future'], away_team_frame_df['y_future']])
        ball_true_scatter.set_offsets(np.c_[ball_frame_df['x_future'], ball_frame_df['y_future']])

        home_team_pred_scatter.set_offsets(np.c_[home_team_frame_df['x_future_pred'], home_team_frame_df['y_future_pred']])
        away_team_pred_scatter.set_offsets(np.c_[away_team_frame_df['x_future_pred'], away_team_frame_df['y_future_pred']])
        ball_pred_scatter.set_offsets(np.c_[ball_frame_df['x_future_pred'], ball_frame_df['y_future_pred']])

        if draw_prediction_lines:
            home_team_error_lines.set_segments([[[row['x_future'], row['y_future']], [row['x_future_pred'], row['y_future_pred']]] for _, row in home_team_frame_df.iterrows()])
            away_team_error_lines.set_segments([[[row['x_future'], row['y_future']], [row['x_future_pred'], row['y_future_pred']]] for _, row in away_team_frame_df.iterrows()])
            ball_error_lines.set_segments([[[row['x_future'], row['y_future']], [row['x_future_pred'], row['y_future_pred']]] for _, row in ball_frame_df.iterrows()])

        frame_text.set_text(f"{home_team_frame_df.iloc[0]['minute']}:{home_team_frame_df.iloc[0]['second']}:{home_team_frame_df.iloc[0]['frame'] % FPS}")
        pred_error = frames_df[frames_df['frame'] == frame]['pred_error'].mean()
        if pred_error is not None:
            error_text.set_text(f"Error: {round(pred_error, 2)} m")

    animation = FuncAnimation(fig, update, frames=range(start_frame, end_frame + 1), interval=200)

    # Collect frames for GIF creation
    frames = []
    for frame in range(start_frame, end_frame + 1):
        update(frame)
        fig.canvas.draw()  # Update canvas
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    # Save frames as GIF
    gif_path = f"animations/{home_team}_vs_{away_team}_{model_name}_frames_{start_frame}-{end_frame}.gif"
    imageio.mimsave(gif_path, frames, fps=8)

# Visualize the result from a training session
def visualize_training_results(training_results, model_name):
    epochs = range(1, len(training_results['loss']) + 1)

    plt.figure(facecolor='white')
    plt.plot(epochs, training_results['loss'], 'b', color=ajax_red, label='Training loss')
    plt.plot(epochs, training_results['val_loss'], 'r', color=ajax_yellow, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if len(epochs) < 10:
        plt.xticks(np.arange(1, len(epochs) + 1, 1.0))  # Set x_ticks to integers
    else:
        plt.xticks(np.arange(2, len(epochs) + 1, len(epochs) / 10))  # Set x_ticks to integers
    plt.xlim(1, len(epochs))  # Set start_x and end_x
    plt.grid(True, alpha=0.3)  # Add grid with less visibility
    plt.legend()

    # Save figure
    save_path = f"images/models/training_results_{model_name}.png"
    plt.savefig(save_path)