import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# --- Setup paths ---
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.append(str(SRC_DIR))

# --- Load environment variables ---
load_dotenv(BASE_DIR / ".env")

# --- Imports ---
from articles import fetch_articles
from nlp_models import HuggingFaceEmbeddings
from utils import get_ground_truth, compare_articles_to_ground_truth

def main():
    # --- Parameters ---
    SEARCH_QUERY = "2024 Champions League Final"
    LANGUAGE = "en"
    TOTAL_ARTICLES = 100
    PAGE_SIZE = 5
    TOPIC_LABEL = "champions+league+2024"
    PUBLISHED_AFTER = "2024-01-01"

    ARTICLES_CSV = OUTPUT_DIR / "articles.csv"
    EMBEDDED_CSV = OUTPUT_DIR / "embedded_articles.csv"
    SCORED_CSV = OUTPUT_DIR / "articles_scored.csv"
    GROUND_TRUTH_TXT = OUTPUT_DIR / "ground_truth.txt"

    # --- Step 1: Fetch articles ---
    print("Fetching articles...")
    articles = fetch_articles(
        query=SEARCH_QUERY,
        language=LANGUAGE,
        total_articles=TOTAL_ARTICLES,
        page_size=PAGE_SIZE,
        topic_label=TOPIC_LABEL,
        published_after=PUBLISHED_AFTER
    )
    df_articles = pd.DataFrame(articles)
    df_articles.to_csv(ARTICLES_CSV, index=False)
    print(f"Saved raw articles to {ARTICLES_CSV}")

    # --- Step 2: Generate text embeddings ---
    print("Generating text embeddings...")
    embedder = HuggingFaceEmbeddings(path=str(ARTICLES_CSV), save_path=str(OUTPUT_DIR))
    embedder.get_embedding_df(column='body', directory=str(OUTPUT_DIR), file=EMBEDDED_CSV.name)
    print(f"Embeddings saved to {EMBEDDED_CSV}")

    # --- Step 3: Extract ground truth from all articles ---
    print("Extracting ground truth...")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    ground_truth = get_ground_truth(str(ARTICLES_CSV), nlp)

    with open(GROUND_TRUTH_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(ground_truth))
    print(f"Ground truth written to {GROUND_TRUTH_TXT}")

    # --- Step 4: Compare articles to ground truth ---
    print("Comparing articles to ground truth...")
    df_scored = compare_articles_to_ground_truth(df_articles, ground_truth, nlp)
    df_scored.to_csv(SCORED_CSV, index=False)
    print(f"Similarity scores saved to {SCORED_CSV}")

if __name__ == "__main__":
    main()
