# Part 1: Project Setup and Data Loading

import pandas as pd
import os

# Get the directory where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to data files (going one level up from 'src' to the root)
train_path = os.path.join(base_dir, '..', 'data', 'spam_train_10000.csv')
test_path = os.path.join(base_dir, '..', 'data', 'spam_test_1000.csv')


def load_datasets():
    """
    Loads the train and test CSV files and returns them as DataFrames.
    """
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Datasets loaded successfully!")
        return train_df, test_df
    else:
        print(f"Error: Dataset files not found at: {os.path.abspath(os.path.join(base_dir, '..', 'data'))}")
        return None, None


