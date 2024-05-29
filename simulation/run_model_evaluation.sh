#!/bin/bash
# Convert Jupyter Notebook to Python script
jupyter nbconvert --to script model-evaluation.ipynb

# Execute the converted Python script
python3 model-evaluation.py

# Delete the temporary file
rm model-evaluation.py