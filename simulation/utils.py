# File with utility functions
from io import StringIO
import pandas as pd
import requests

# Fetches data from a Google Sheets CSV URL and converts it into a DataFrame
def google_sheet_to_df(url):
    # Fetch the CSV data from the URL
    response = requests.get(url)
    response.raise_for_status()

    # Decode the response text
    response.encoding = 'utf-8'

    # Read the CSV data into a pandas DataFrame
    data = StringIO(response.text)
    data_df = pd.read_csv(data)
    return data_df

# Load the processed/frames
def load_processed_frames(n_matches=None, match_id=None):
    # Create DataFrame for storing all frames
    frames_dfs = []
    # Load frames_df
    for selected_season in seasons:
        for selected_competition in competitions:
            # Define paths
            DATA_FOLDER_PROCESSED = f"{DATA_LOCAL_FOLDER}/data/{selected_season}/{selected_competition}/processed"

            # Find all frames parquet files
            match_paths = glob.glob(os.path.join(DATA_FOLDER_PROCESSED, "*.parquet"))

            # Extract the IDs without the ".parquet" extension
            if match_id >= 0:
                # Only load the match with the given match_id
                match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths][match_id : match_id + 1]
            elif n_matches > 0:
                # Only load the specified number of matches
                match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths][0:n_matches]
            else:
                # Load all matches
                match_ids = [os.path.splitext(os.path.basename(path))[0] for path in match_paths]

            # For all matches
            for match_id in match_ids:
                # Convert parquet file to a DataFrame
                file_path_match = f"{DATA_FOLDER_PROCESSED}/{match_id}.parquet"
                frames_df = pd.read_parquet(file_path_match)
                
                # Append the DataFrame to frames_dfs
                frames_dfs.append(frames_df)

    return frames_dfs