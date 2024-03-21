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