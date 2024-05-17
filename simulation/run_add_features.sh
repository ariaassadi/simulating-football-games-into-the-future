#!/bin/bash
# Convert Jupyter Notebook to Python script
jupyter nbconvert --to script add-features-frames-df.ipynb

# Execute the converted Python script
python3 add-features-frames-df.py

# Delete the temporary file
rm add-features-frames-df.py