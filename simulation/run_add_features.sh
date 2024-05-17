#!/bin/bash
# Convert Jupyter Notebook to Python script
jupyter nbconvert --to script add-features.ipynb

# Execute the converted Python script
python3 add-features.py

# Delete the temporary file
rm add-features.py