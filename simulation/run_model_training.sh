#!/bin/bash
# Convert Jupyter Notebook to Python script
jupyter nbconvert --to script model-training.ipynb

# Execute the converted Python script
python3 model-training.py