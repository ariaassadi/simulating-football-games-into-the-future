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

print(long_to_short_name['BK Häcken'])
path_to_db = path + '/data_processing' + '/data' + '/2sec.sqlite'


schedule_df = pd.DataFrame(columns=['match_id', 'home_team_name', 'away_team_name', 'home_team_name_short', 'away_team_name_short'])
write_to_db = False

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
    
    new_df = pd.DataFrame({'match_id': match_id, 'home_team_name': home_team_name, 'away_team_name': away_team_name, 'home_team_name_short': long_to_short_name[home_team_name], 'away_team_name_short': long_to_short_name[away_team_name]}, index=[i])
    
    # concat the new_df to the schedule_df
    schedule_df = pd.concat([schedule_df, new_df])
    
print(schedule_df)

write_to_db = True

if True:
    print('Writing to DB' + path_to_db)
    # Connect to the SQLite database
    conn = sqlite3.connect(path_to_db)

    table_name = f'schedule'
    schedule_df.to_sql(table_name, conn, if_exists='replace')

    # Close the connection
    conn.close()