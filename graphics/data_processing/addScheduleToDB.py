import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

import sqlite3

path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir))
data_path = path + '/data_processing' + '/data' + '/games'
# print(data_path)
match_paths = glob.glob(os.path.join(data_path, "*.parquet"))
match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths]

# Dictionary to convert long name to short name:
df_club_info = pd.read_excel(path + '/data_processing' + '/data' + '/club_info.xlsx')

long_to_short_name = {item['Team']: item['Short Name'] for item in df_club_info[["Team", "Short Name"]].to_dict(orient='records')}

home_colors_dict = {item['Team']: item['Home Color Hex'] for item in df_club_info[["Team", "Home Color Hex"]].to_dict(orient='records')}
away_colors_dict = {item['Team']: item['Away Color Hex'] for item in df_club_info[["Team", "Away Color Hex"]].to_dict(orient='records')}

print(long_to_short_name['BK Häcken'])
path_to_db = path + '/data_processing' + '/data' + '/2sec.sqlite'


schedule_df = pd.DataFrame(columns=['match_id', 'home_team_name', 'away_team_name', 'home_team_name_short', 'away_team_name_short'])
write_to_db = False

def hex_to_rgb(hex_code):
    # Convert hexadecimal color code to RGB tuple
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def color_distance(color1, color2):
    # Calculate Euclidean distance between two colors in RGB space
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5

def select_away_team_color(home_team_color_hex, color_option1_hex, color_option2_hex, threshold=100):
    # Convert hexadecimal color codes to RGB tuples
    home_team_color = hex_to_rgb(home_team_color_hex)
    color_option1 = hex_to_rgb(color_option1_hex)
    color_option2 = hex_to_rgb(color_option2_hex)
    
    # Check if the home team's color is too similar to color_option1
    if color_distance(home_team_color, color_option1) < threshold:
        return color_option2_hex
    else:
        return color_option1_hex


for i, match_id in enumerate(match_ids):
    file_path_match = f"{data_path}/{match_id}.parquet"
    print(file_path_match)
    frames_df = pd.read_parquet(file_path_match)
    frames_df.drop(columns=['minute', 'second', 'ball_in_motion', 'a_x', 'a_y', 'role', 'distance_ran', 'events'], inplace=True)
    if write_to_db:
        # Connect to the SQLite database
        conn = sqlite3.connect(path_to_db)

        # Iterate over each DataFrame in frames_dfs and write it to the database

        if (i==0):
            table_name = f'games'
            frames_df.to_sql(table_name, conn, if_exists='replace')
        else:
            table_name = f'games'
            frames_df.to_sql(table_name, conn, if_exists='append')

        # Close the connection
        conn.close()
    frames_df['team_name'] = frames_df['team_name'].astype(str)    
    
    home_team_df = frames_df[frames_df['team'] == 'home_team']
    away_team_df = frames_df[frames_df['team'] == 'away_team']
    
    match_id = frames_df['match_id'].iloc[0]
    home_team_name = home_team_df['team_name'].iloc[0]
    away_team_name = away_team_df['team_name'].iloc[0]
    
    home_team_color = home_colors_dict[home_team_name]
    away_team_color = select_away_team_color(home_team_color, home_colors_dict[away_team_name], away_colors_dict[away_team_name])
    
    
    
    new_df = pd.DataFrame({'match_id': match_id,
                           'home_team_name': home_team_name,
                           'away_team_name': away_team_name,
                           'home_team_name_short': long_to_short_name[home_team_name],
                           'away_team_name_short': long_to_short_name[away_team_name],
                           'home_team_color': home_team_color,
                           'away_team_color': away_team_color},
                          index=[i])
    
    # concat the new_df to the schedule_df
    schedule_df = pd.concat([schedule_df, new_df])
    
print(schedule_df)

write_to_db = True

if write_to_db:
    print('Writing to DB' + path_to_db)
    # Connect to the SQLite database
    conn = sqlite3.connect(path_to_db)

    table_name = f'schedule'
    schedule_df.to_sql(table_name, conn, if_exists='replace')

    # Close the connection
    conn.close()