import pandas as pd
import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")



######### Merge Datasets #########
def process_embeddings(df:pd.DataFrame, col_name):
    """
    Process embeddings in a DataFrame column.

    Args:
    - df (pd.DataFrame): The DataFrame containing the embeddings column.
    - col_name (str): The name of the column containing the embeddings.

    Returns:
    pd.DataFrame: The DataFrame with processed embeddings.

    Steps:
    1. Convert the values in the specified column to lists.
    2. Extract values from lists and create new columns for each element.
    3. Remove the original embeddings column.

    Example:
    df_processed = process_embeddings(df, 'embeddings')
    """
    # Convert the values in the column to lists
    df[col_name] = df[col_name].apply(eval)

    # Extract values from lists and create new columns
    embeddings_df = pd.DataFrame(df[col_name].to_list(), columns=[f"text_{i+1}" for i in range(df[col_name].str.len().max())])
    df = pd.concat([df, embeddings_df], axis=1)

    # Remove the original "embeddings" column
    df = df.drop(columns=[col_name])

    return df

# Preprocess and merge the dataframes
def preprocess_data(text_data, image_data, text_id="image_id", image_id="ImageName", embeddings_col = 'embeddings'):
    """
    Preprocess and merge text and image dataframes.

    Args:
    - text_data (pd.DataFrame): DataFrame containing text data.
    - image_data (pd.DataFrame): DataFrame containing image data.
    - text_id (str): Column name for text data identifier.
    - image_id (str): Column name for image data identifier.
    - embeddings_col (str): Column name for embeddings data.

    Returns:
    pd.DataFrame: Merged and preprocessed DataFrame.

    This function:
    Process text and image embeddings.
    Convert image_id and text_id values to integers.
    Merge dataframes using id.
    Drop unnecessary columns.

    Example:
    merged_df = preprocess_data(text_df, image_df)
    """
    text_data = process_embeddings(text_data, embeddings_col)
    image_data = rename_image_embeddings(image_data)    

    # drop missing values in image id
    image_data = image_data.dropna(subset=[image_id])
    text_data = text_data.dropna(subset=[text_id])

    text_data[text_id] = text_data[text_id].apply(lambda x: x.split('/')[-1])
    
    # Merge dataframes using image_id
    df = pd.merge(text_data, image_data, left_on=text_id, right_on=image_id)

    # Drop unnecessary columns
    df.drop([image_id, text_id], axis=1, inplace=True)

    return df
