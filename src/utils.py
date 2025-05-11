import pandas as pd
import spacy
from collections import Counter
import numpy as np

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

def reduced_embeddings(embeddings:list, new_size:int, how:str='rand') -> np.ndarray:
    """
    This method will take in a list of lists containing the embeddings for each
    of the articles, and return a numpy array of the same length were each of the
    embeddings have been reduced in size by the specified method

    Args:
        embeddings (list of numpy.ndarray): A list where each element is a text embedding
        new_size (int): The desired size of the output averaged array.
        how (str) : The way in which the embeddings will be reduced, default is random

    Returns:
        reduced_embds (numpy.ndarray): A 1D NumPy array of size new_size containing the
        averaged information.
    """
    if how == 'rand':
        reduced_embds = np.ndarray((len(embeddings),new_size))

        for i,vector in enumerate(embeddings):
            arr = np.asarray(vector)
            rand_values = np.random.choice(arr,size=new_size)
            reduced_embds[i] = rand_values
        
        return reduced_embds
    print('Not implemented yet!')
    return None
        

######### Details and text utilities #########
def preprocess_text(text: str):
    if not isinstance(text, str):
        return ""  # or return None if you'd rather skip processing
    text = text.lower()
    text = ' '.join(text.split())
    return text


def extract_details(text,nlp_model:spacy.Language):
    """Extracts named entities and key noun phrases."""
    doc = nlp_model(preprocess_text(text))
    details = set()

    # Extract Named Entities (People, Orgs, Locations, Products etc.)
    for ent in doc.ents:
        # Filter entities
        # PERSON, ORG, GPE (Geo-Political Entity), PRODUCT, EVENT, DATE, MONEY
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
             details.add(ent.text.strip()) # Add entity text

    # Extract key concepts/phrases
    for chunk in doc.noun_chunks:
        # Optional: Filter noun chunks (e.g., length, content)
        details.add(chunk.text.strip())

    return details

# get the ground truth from a given csv file containing all the relevant articles
# columns in the csv file are: article_id, title, body, source, published_at, url
def get_ground_truth(articles_path:str, nlp_model) -> list:
    df_articles = pd.read_csv(articles_path)
    articles = df_articles['body'].to_list()
    
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
    threshold = max(2, int(num_articles*0.2))

    # we check each detail and only include in our final list the ones that surpass our threshold
    groun_truth = [detail for detail, count in detail_counts.items() if count >= threshold]

    print("\n--- Common Details Found ---")
    if groun_truth:
        for detail in groun_truth:
            print(f"- {detail} (Found in {detail_counts[detail]} articles)")
    else:
        print("No common details found based on the threshold.")
    
    return groun_truth

def compare_articles_to_ground_truth(df_articles:pd.DataFrame, ground_truth:np.ndarray) -> pd.DataFrame:
    """
    Compares each article's extracted details to the ground truth list, and returns a new dataframe containing the
    cosine similarity score between each article's embedding and the hard truth embedding

    Args:
        df_articles (pd.DataFrame): DataFrame with a 'embeddings' column.
        ground_truth (np.ndarray): Array of text embeddings.

    Returns:
        pd.DataFrame: Original DataFrame with a new 'cosine_similarity_score' column.
    """

    def cosine_similarity(vec1, vec2):
        dot_prod = np.dot(vec1,vec2)
        magnitude_1 = np.linalg.norm(vec1)
        magnitude_2 = np.linalg.norm(vec2)
        if magnitude_1 == 0 or magnitude_2 == 0:
            return 0
        return dot_prod / (magnitude_1 * magnitude_2)

    similarity_scores = []

    articles_embeddings = df_articles['embeddings'].apply(eval)
    articles_embeddings = articles_embeddings.to_list()

    for emb in articles_embeddings:
        score = cosine_similarity(np.asarray(emb),np.asarray(ground_truth))
        similarity_scores.append(score)

    df_articles['similarity_to_ground_truth'] = similarity_scores
    return df_articles
