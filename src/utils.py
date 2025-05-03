import pandas as pd
import numpy as np
from collections import Counter

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


######### Details and text utilities #########
def preprocess_text(text:str):
    text = text.lower()

    # remove whitespace
    text = ' '.join(text.split())

    return text

def extract_details(text,nlp_model):
    """Extracts named entities and key noun phrases."""
    doc = nlp_model(preprocess_text(text))
    details = set() # Use a set to avoid duplicates within the same article

    # Extract Named Entities (People, Orgs, Locations, Products etc.)
    for ent in doc.ents:
        # Filter entities
        # Common types: PERSON, ORG, GPE (Geo-Political Entity), PRODUCT, EVENT, DATE, MONEY
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
             details.add(ent.text.strip()) # Add entity text

    # Extract key concepts/phrases
    for chunk in doc.noun_chunks:
        # Optional: Filter noun chunks (e.g., length, content)
        details.add(chunk.text.strip())

    return details

def get_ground_truth(articles:list, nlp_model) -> list:
    all_details = list()

    for i, article_text in enumerate(articles):
        print(f"----------Processing Article {i+1}----------")
        extracted = extract_details(article_text, nlp_model)
        all_details.extend(list(extracted))
    # this counts how many times each detail appear
    detail_counts = Counter(all_details)

    # threshold at which a detail must appear throughout all
    # the articles to be considered part of the ground truth
    # min of 2 articles, max is set as a fracction of the total number of articles
    num_articles = len(articles)
    threshold = max(2, int(num_articles*0.75))

    # we check each detail and only include in our final list the ones that surpass our threshold
    groun_truth = [detail for detail, count in detail_counts.items() if count >= threshold]

    print("\n--- Common Details Found ---")
    if groun_truth:
        for detail in groun_truth:
            print(f"- {detail} (Found in {detail_counts[detail]} articles)")
    else:
        print("No common details found based on the threshold.")
    
    return groun_truth