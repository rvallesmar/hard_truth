""" Evaluate Medical Tests Classification in LLMS """
## Setup
#### Load the API key and libaries.
import os

import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch


## Hugging face Models
class HuggingFaceEmbeddings:
    """
    A class to handle text embedding generation using a Hugging Face pre-trained transformer model.
    This class loads the model, tokenizes the input text, generates embeddings, and provides an option 
    to save the embeddings to a CSV file.

    Args:
        model_name (str, optional): The name of the Hugging Face pre-trained model to use for generating embeddings. 
                                    Default is 'sentence-transformers/all-MiniLM-L6-v2'.
        path (str, optional): The path to the CSV file containing the text data. Default is 'data/file.csv'.
        save_path (str, optional): The directory path where the embeddings will be saved. Default is 'data'.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). If None, it will automatically detect 
                                a GPU if available; otherwise, it defaults to CPU.

    Attributes:
        model_name (str): The name of the Hugging Face model used for embedding generation.
        tokenizer (transformers.AutoTokenizer): The tokenizer corresponding to the chosen model.
        model (transformers.AutoModel): The pre-trained model loaded for embedding generation.
        path (str): Path to the input CSV file.
        save_path (str): Directory where the embeddings CSV will be saved.
        device (torch.device): The device on which the model and data are processed (CPU or GPU).

    Methods:
        get_embedding(text):
            Generates embeddings for a given text input using the pre-trained model.

        get_embedding_df(column, directory, file):
            Reads a CSV file, computes embeddings for a specified text column, and saves the resulting DataFrame 
            with embeddings to a new CSV file in the specified directory.

    Example:
        embedding_instance = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                                   path='data/products.csv', save_path='output')
        text_embedding = embedding_instance.get_embedding("Sample product description.")
        embedding_instance.get_embedding_df(column='description', directory='output', file='product_embeddings.csv')
    """
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', path='data/file.csv', save_path='data', device=None):
        """
        Initializes the HuggingFaceEmbeddings class with the specified model and paths.

        Args:
            model_name (str, optional): The name of the Hugging Face pre-trained model. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
            path (str, optional): The path to the CSV file containing text data. Default is 'data/file.csv'.
            save_path (str, optional): Directory path where the embeddings will be saved. Default is 'Models'.
            device (str, optional): Device to use for model processing. Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        self.model_name = model_name
        # Load the Hugging Face tokenizer from a pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Load the model from the Hugging Face model hub from the specified model name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.path = path
        self.save_path = save_path or 'Models'
        
        # Define device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        
        # Move model to the specified device
        self.model.to(self.device)
        print(f"Model moved to device: {self.device}")
        print(f"Model: {model_name}")
        
    def get_embedding(self, text):
        """
        Generates embeddings for a given text using the Hugging Face model.

        Args:
            text (str): The input text for which embeddings will be generated.

        Returns:
            list: A list containing the embedding vector for the input text.
        """
        ### Tokenize the input text using the Hugging Face tokenizer
        inputs = self.tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=512)
        
        # Move the inputs to the device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            # Generate the embeddings using the Hugging Face model from the tokenized input
            outputs = self.model(**inputs)
        
        # Extract the embeddings from the model output, send to cpu and return the list
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        
        return embeddings

    def get_embedding_df(self, column, directory, file):
        output_dir = directory or self.save_path
        output_path = os.path.join(output_dir,file)
        # Load the CSV file
        df = pd.read_csv(self.path)
        # Generate embeddings for the specified column using the `get_embedding` method
        # convert the embeddings to a list before saving to the DataFrame
        df["embeddings"] = df[column].apply(lambda x: self.get_embedding(x) if isinstance(x, str) else None)


        df = df.dropna(subset=['embeddings'])
        
        os.makedirs(directory, exist_ok=True)
        # Save the DataFrame with the embeddings to a new CSV file in the specified directory
        df.to_csv(output_path, index=False)

        return df
    
    @staticmethod
    def get_single_embedding(text, model='sentence-transformers/all-MiniLM-L6-v2', device = None):
        """
        Generates embeddings for a given text using the specified huggingface model.

        Args:
            text (str): The input text for which embeddings will be generated.

        Returns:
            list: A list containing the embedding vector for the input text.
        """
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)

        # Define device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        # Move model to the specified device
        model.to(device)

        # Tokenize the input text
        inputs = tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=512)        
        # Move the inputs to the device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            # Generate the embeddings using the Hugging Face model from the tokenized input
            outputs = model(**inputs)
        # Extract the embeddings from the model output, send to cpu and return the list
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        
        return embeddings

