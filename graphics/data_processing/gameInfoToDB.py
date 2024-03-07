import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

import sqlite3

path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir))
data_path = path + '/data_processing' + '/data' + '/game_info'
# print(data_path)
match_paths = glob.glob(os.path.join(data_path, "*.json"))
match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths]

frames_dfs = []

for match_id in match_ids:
    file_path_match = f"{data_path}/{match_id}.json"
    print(file_path_match)
    frames_df = pd.read_parquet(file_path_match)
    frames_dfs.append(frames_df)
    
for i in range(len(frames_dfs)):
    frames_dfs[i].drop(columns=['minute', 'second', 'ball_in_motion', 'a_x', 'a_y', 'role', 'distance_ran', 'events'], inplace=True)


path_to_db = path + '/data_processing' + '/data' + '/2sec.sqlite'
# Connect to the SQLite database
conn = sqlite3.connect(path_to_db)

# Iterate over each DataFrame in frames_dfs and write it to the database
for i, df in enumerate(frames_dfs):
    table_name = f'game_info_table'
    df.to_sql(table_name, conn, if_exists='append')

# Close the database connection
conn.close()