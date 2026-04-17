# Part 1: Project Setup and Data Loading

import pandas as pd
from sklearn.model_selection import train_test_split

def load_datasets():
    """
    Loads the train and test CSV files and returns them as DataFrames.
    """
    df = pd.read_csv("../data/Spam_SMS.csv")
    df = df.rename(columns={'Message': 'text', 'Class': 'label'})
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    train_df, test_df=train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df
