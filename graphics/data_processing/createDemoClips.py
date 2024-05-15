import pandas as pd
import numpy as np
import os
import glob
import json
from pathlib import Path

import sqlite3

path = os.getcwd()
data_path = path + '/data' + '/demo_data'
# print(data_path)S
match_paths = glob.glob(os.path.join(data_path, "*.parquet"))
match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths]

# Dictionary to convert long name to short name:
df_club_info = pd.read_excel(path + '/data' + '/club_info.xlsx')

long_to_short_name = {item['Team']: item['Short Name'] for item in df_club_info[["Team", "Short Name"]].to_dict(orient='records')}

home_colors_dict = {item['Team']: item['Home Color Hex'] for item in df_club_info[["Team", "Home Color Hex"]].to_dict(orient='records')}
away_colors_dict = {item['Team']: item['Away Color Hex'] for item in df_club_info[["Team", "Away Color Hex"]].to_dict(orient='records')}

path_to_db = path + '/data' + '/2sec_demo_clips.sqlite'


schedule_df = pd.DataFrame(columns=['match_id', 'home_team_name', 'away_team_name', 'home_team_name_short', 'away_team_name_short'])
write_to_db = True

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

# Malmö FF vs BK Häcken: 
# match_id = a641b1a0-0603-4a57-81e4-2cbc188ab05c
# frame for each clip respectively 0-710, 10500-11893, 24256-25772

# IF Elfsborg vs Djurgårdens IF:
# match_id = 07313d0f-7410-465f-b97b-d5e980a172ff
# frame for each clip respectively 0-1496, 18550-20074

match_clips = {'a641b1a0-0603-4a57-81e4-2cbc188ab05c':[(69085,69795), (79585, 80978), (93341,94857)], '07313d0f-7410-465f-b97b-d5e980a172ff':[(72255,73751), (90805,92329)]}
i = 0
match_id = 'a641b1a0-0603-4a57-81e4-2cbc188ab05c'
file_path_match = f"{data_path}/{match_id}.parquet"
# print(file_path_match)
frames_df = pd.read_parquet(file_path_match)
frames_df.drop(columns=['minute', 'second', 'ball_in_motion', 'a_x', 'a_y', 'role', 'distance_ran', 'events'], inplace=True)
print(frames_df[frames_df['period'] == 1]['frame'].tail(1))
print(frames_df[frames_df['period'] == 1]['frame'].head(1))


for match_id in match_clips:
    file_path_match = f"{data_path}/{match_id}.parquet"
    # print(file_path_match)
    frames_df = pd.read_parquet(file_path_match)
    frames_df.drop(columns=['minute', 'second', 'ball_in_motion', 'a_x', 'a_y', 'role', 'distance_ran', 'events'], inplace=True)
    print(frames_df[frames_df['period'] == 2]['frame'].head(1))
    for (start_frame, end_frame) in match_clips[match_id]:
        # filter based on start_frame, end_frame and period
        filtered_df = frames_df[(frames_df['frame'] >= start_frame) & (frames_df['frame'] <= end_frame)]
        # filtered_df = filtered_df[filtered_df['period'] == period]
        # print(filtered_df.info())

        # write to json
        file_name = f'{match_id}_{start_frame}_{end_frame}.json'
        formated_json = {'game': filtered_df.to_dict(orient='records')}
        with open(f'{path}/data/clips/{file_name}', 'w') as f:
            json.dump(formated_json, f, indent=2)

        # write to db
        # if write_to_db:
        #     # Connect to the SQLite database
        #     conn = sqlite3.connect(path_to_db)

        #     # Iterate over each DataFrame in frames_dfs and write it to the database

        #     if (i==0):
        #         table_name = f'games'
        #         frames_df.to_sql(table_name, conn, if_exists='replace')
        #     else:
        #         table_name = f'games'
        #         frames_df.to_sql(table_name, conn, if_exists='append')

        #     # Close the connection
        #     conn.close()

        filtered_df['team_name'] = filtered_df['team_name'].astype(str)    
    
        home_team_df = filtered_df[frames_df['team'] == 'home_team']
        away_team_df = filtered_df[frames_df['team'] == 'away_team']
        
        match_id = filtered_df['match_id'].iloc[0]
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
                            'away_team_color': away_team_color,
                            'clip': file_name},
                            index=[i])
        i += 1
        # concat the new_df to the schedule_df
        schedule_df = pd.concat([schedule_df, new_df])
        
filtered_df = frames_df[frames_df['period'] == 1]

print(filtered_df.info())

# write to json
start_frame = 0
end_frame = frames_df[frames_df['period'] == 1]['frame'].tail(1).values[0]
file_name = f'{match_id}_{start_frame}_{end_frame}.json'
formated_json = {'game': filtered_df.to_dict(orient='records')}
with open(f'{path}/data/clips/{file_name}', 'w') as f:
    json.dump(formated_json, f, indent=2)


# write to db
# if write_to_db:
#     # Connect to the SQLite database
#     conn = sqlite3.connect(path_to_db)

#     # Iterate over each DataFrame in frames_dfs and write it to the database

#     if (i==0):
#         table_name = f'games'
#         frames_df.to_sql(table_name, conn, if_exists='replace')
#     else:
#         table_name = f'games'
#         frames_df.to_sql(table_name, conn, if_exists='append')

#     # Close the connection
#     conn.close()

filtered_df['team_name'] = filtered_df['team_name'].astype(str)    

home_team_df = filtered_df[frames_df['team'] == 'home_team']
away_team_df = filtered_df[frames_df['team'] == 'away_team']

match_id = filtered_df['match_id'].iloc[0]
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
                    'away_team_color': away_team_color,
                    'clip': file_name},
                    index=[i])
# concat the new_df to the schedule_df
schedule_df = pd.concat([schedule_df, new_df])

write_to_db = True

if write_to_db:
    print('Writing to DB' + path_to_db)
    # Connect to the SQLite database
    conn = sqlite3.connect(path_to_db)

    table_name = f'schedule'
    schedule_df.to_sql(table_name, conn, if_exists='replace')

    # Close the connection
    conn.close()