import pandas as pd
import sys
import os
import time

def main(output_path):
    # Define the array you want to write to JSON
    data = {
        'pitch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11],
    }

    # Start timing the creation of the DataFrame
    start_time = time.time()
    
    # Create a DataFrame from the data
    sr = pd.Series(data)
    print(sr)
    
    # End timing the creation of the DataFrame
    creation_time = time.time() - start_time
    print("Time taken to create DataFrame:", creation_time, "seconds")

    # Define the path to the JSON file
    path = os.path.abspath(output_path)

    # Start timing writing DataFrame to JSON
    start_time = time.time()
    
    # Write the DataFrame to JSON
    sr.to_json(path)
    
    # End timing writing DataFrame to JSON
    writing_time = time.time() - start_time
    print("Time taken to write DataFrame to JSON:", writing_time, "seconds")
    print("JSON file written successfully to:", path)

    # Start timing reading JSON file back into a Series
    start_time = time.time()
    
    # Read the JSON file back into a Series
    accy = pd.read_json(path, typ='series')
    
    # End timing reading JSON file back into a Series
    reading_time = time.time() - start_time
    print("Time taken to read JSON file back into a Series:", reading_time, "seconds")
    print("Data read back from JSON file:")
    print(accy)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hello_world.py <output_path>")
        sys.exit(1)
    
    output_path = sys.argv[1]
    main(output_path)
