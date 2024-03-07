from collections import OrderedDict
from datetime import datetime
import pandas as pd
import random
import json
import glob
import os

from settings import *

# Define Allsvenskan schedule for the 2023 season
allsvenskan_schedule_2022 = {
    '1': (datetime(2022, 4, 2,), datetime(2022, 4, 5,)),
    '2': (datetime(2022, 4, 9,), datetime(2022, 4, 12,)),
    '3': (datetime(2022, 4, 15,), datetime(2022, 4, 18,)),
    '4': (datetime(2022, 4, 20,), datetime(2022, 4, 22,)),
    '5': (datetime(2022, 4, 24,), datetime(2022, 4, 27,)),
    '6': (datetime(2022, 4, 30,), datetime(2022, 5, 2,)),
    '7': (datetime(2022, 5, 7,), datetime(2022, 5, 10,)),
    '8': (datetime(2022, 5, 14,), datetime(2022, 5, 17,)),
    '9': (datetime(2022, 5, 21,), datetime(2022, 5, 24,)),
    '10': (datetime(2022, 5, 28,), datetime(2022, 5, 30,)),
    '11': (datetime(2022, 6, 26,), datetime(2022, 6, 28,)),
    '12': (datetime(2022, 7, 1,), datetime(2022, 7, 4,)),
    '13': (datetime(2022, 7, 9,), datetime(2022, 7, 12,)),
    '14': (datetime(2022, 7, 16,), datetime(2022, 7, 19,)),
    '15': (datetime(2022, 7, 23,), datetime(2022, 7, 26,)),
    '16': (datetime(2022, 5, 5,), datetime(2022, 8, 2,)), # Special gameweek
    '17': (datetime(2022, 8, 6,), datetime(2022, 8, 9,)),
    '18': (datetime(2022, 8, 13), datetime(2022, 8, 16)),
    '19': (datetime(2022, 8, 20,), datetime(2022, 8, 23,)),
    '20': (datetime(2022, 8, 27,), datetime(2022, 8, 30,)),
    '21': (datetime(2022, 9, 4,), datetime(2022, 9, 6,)),
    '22': (datetime(2022, 9, 10,), datetime(2022, 9, 13,)),
    '23': (datetime(2022, 9, 17,), datetime(2022, 9, 19,)),
    '24': (datetime(2022, 10, 1,), datetime(2022, 10, 4,)),
    '25': (datetime(2022, 10, 8,), datetime(2022, 10, 11,)),
    '26': (datetime(2022, 10, 14,), datetime(2022, 10, 17,)),
    '27': (datetime(2022, 10, 19,), datetime(2022, 10, 21,)),
    '28': (datetime(2022, 10, 23,), datetime(2022, 10, 25,)),
    '29': (datetime(2022, 10, 29,), datetime(2022, 11, 1,)),
    '30': (datetime(2022, 11, 6,), datetime(2022, 11, 7,))
  }

# Define Allsvenskan schedule for the 2023 season
allsvenskan_schedule_2023 = {
    '1': (datetime(2023, 4, 1,), datetime(2023, 4, 7,)),
    '2': (datetime(2023, 4, 8,), datetime(2023, 4, 14,)),
    '3': (datetime(2023, 4, 15,), datetime(2023, 4, 21,)),
    '4': (datetime(2023, 4, 22,), datetime(2023, 4, 28,)),
    '5': (datetime(2023, 4, 29,), datetime(2023, 5, 2,)),
    '6': (datetime(2023, 5, 3,), datetime(2023, 5, 6,)),
    '7': (datetime(2023, 5, 7,), datetime(2023, 5, 12,)),
    '8': (datetime(2023, 5, 13,), datetime(2023, 5, 16,)),
    '9': (datetime(2023, 5, 20,), datetime(2023, 5, 23,)),
    '10': (datetime(2023, 5, 27,), datetime(2023, 6, 2,)),
    '11': (datetime(2023, 6, 3,), datetime(2023, 6, 8,)),
    '12': (datetime(2023, 6, 9,), datetime(2023, 6, 12,)),
    '13': (datetime(2023, 7, 1,), datetime(2023, 7, 7,)),
    '14': (datetime(2023, 7, 8,), datetime(2023, 7, 14,)),
    '15': (datetime(2023, 7, 15,), datetime(2023, 7, 20,)),
    '16': (datetime(2023, 7, 21,), datetime(2023, 7, 28,)),
    '17': (datetime(2023, 7, 29,), datetime(2023, 7, 31,)),
    '18': (datetime(2023, 5, 24), datetime(2023, 8, 8)), # Special gameweek
    '19': (datetime(2023, 8, 12,), datetime(2023, 8, 15,)),
    '20': (datetime(2023, 8, 19,), datetime(2023, 8, 22,)),
    '21': (datetime(2023, 8, 27,), datetime(2023, 8, 29,)),
    '22': (datetime(2023, 9, 2,), datetime(2023, 9, 4,)),
    '23': (datetime(2023, 9, 16,), datetime(2023, 9, 19,)),
    '24': (datetime(2023, 9, 23,), datetime(2023, 9, 26,)),
    '25': (datetime(2023, 9, 30,), datetime(2023, 10, 3,)),
    '26': (datetime(2023, 10, 7,), datetime(2023, 10, 9,)),
    '27': (datetime(2023, 10, 21,), datetime(2023, 10, 24,)),
    '28': (datetime(2023, 10, 28,), datetime(2023, 10, 31,)),
    '29': (datetime(2023, 11, 4,), datetime(2023, 11, 7,)),
    '30': (datetime(2023, 11, 12,), datetime(2023, 11, 13,))
  }

# Construct a dictionary to map each year to the correct schedule
allsvenskan_schedule = {
    2022: allsvenskan_schedule_2022,
    2023: allsvenskan_schedule_2023
}

# Function to determine gameweek
def get_gameweek(match_date, year):
    schedule = allsvenskan_schedule[year]
    for gameweek, (start_date, end_date) in schedule.items():
        if start_date <= match_date <= end_date:
            return gameweek
    return match_date

# Store a parquet file with match info for each season in each competition
for selected_season in seasons:
    for selected_competition in competitions:
        DATA_FOLDER_UNPROCESSED = f"{DATA_LOCAL_FOLDER}/signality/{selected_season}/{selected_competition}/"
        OUTPUT_FOLDER = f"{DATA_LOCAL_FOLDER}/data/match_info/"

        # Define lists to store the extracted data
        match_ids = []
        team_home_names = []
        team_away_names = []
        utc_time = []
        gameweeks = []

        # Find all frames parquet files
        match_paths = glob.glob(os.path.join(DATA_FOLDER_UNPROCESSED, "*.json"))

        # Iterate over each JSON file
        for json_file in match_paths:
            # Load JSON data
            with open(json_file, 'r') as f:
                data = json.load(f, object_pairs_hook=OrderedDict)

            # Find the gameweek
            match_date_str = data.get('time_start').split('T')[0]
            match_date = datetime.strptime(match_date_str, '%Y-%m-%d')
            gameweek = get_gameweek(match_date, selected_season)

            # Append results if the competition is correct
            if data.get('competition') == selected_competition:
                # Append the relevant match info
                match_ids.append(data.get('id'))
                team_home_names.append(data.get('team_home_name'))
                team_away_names.append(data.get('team_away_name'))
                utc_time.append(match_date_str)
                gameweeks.append(gameweek)

        # Create a DataFrame from the lists
        match_data_df = pd.DataFrame({
            'match_ID': match_ids,
            'team_home_name': team_home_names,
            'team_away_name': team_away_names,
            'utc_time': utc_time,
            'gameweek': gameweeks
        })
        
        # Convert all columns to the correct type
        match_data_df[['gameweek']] =  match_data_df[['gameweek']].astype('int8')
        match_data_df[['match_ID', 'team_home_name', 'team_away_name', 'utc_time']] =  match_data_df[['match_ID', 'team_home_name', 'team_away_name', 'utc_time']].astype(str)

        # Sort the DataFrame by 'UTC_time'
        match_data_df = match_data_df.sort_values(by=['gameweek', 'utc_time'], ascending=[True, True])

        # Define the output file path
        output_file_path = os.path.join(OUTPUT_FOLDER, f"match_info_{selected_season}_{selected_competition}.parquet")

        # Convert the DataFrame to an parquet file
        match_data_df.reset_index(drop=True)
        match_data_df.to_parquet(output_file_path)