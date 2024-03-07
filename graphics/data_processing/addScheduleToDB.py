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

frames_dfs = []

for match_id in match_ids:
    file_path_match = f"{data_path}/{match_id}.parquet"
    print(file_path_match)
    frames_df = pd.read_parquet(file_path_match)
    frames_dfs.append(frames_df)
    
for i in range(len(frames_dfs)):
    frames_dfs[i].drop(columns=['minute', 'second', 'ball_in_motion', 'a_x', 'a_y', 'role', 'distance_ran', 'events'], inplace=True)

schedule_df = []



for i, frames_df in enumerate(frames_dfs):
    frames_df['team_name'] = frames_df['team_name'].astype(str)    
    
    home_team_df = frames_df[frames_df['team'] == 'home_team']
    away_team_df = frames_df[frames_df['team'] == 'away_team']
    
    match_id = frames_df['match_id'].iloc[0]
    home_team_name = home_team_df['team_name'].iloc[0]
    away_team_name = away_team_df['team_name'].iloc[0]
    
    schedule_df.append(pd.DataFrame({'match_id': match_id, 'home_team_name': home_team_name, 'away_team_name': away_team_name}, index=[i]))
print(schedule_df[0])

write_to_db = False

if write_to_db:
    path_to_db = path + '/data_processing' + '/data' + '/2sec.sqlite'
    # Connect to the SQLite database
    conn = sqlite3.connect(path_to_db)

    # Iterate over each DataFrame in frames_dfs and write it to the database
    for i, df in enumerate(frames_dfs):
        if (i==0):
            table_name = f'games'
            df.to_sql(table_name, conn, if_exists='replace')
        else:
            table_name = f'games'
            df.to_sql(table_name, conn, if_exists='append')

    # Close the connection
    conn.close()