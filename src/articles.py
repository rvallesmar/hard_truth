import requests
import pandas as pd
import os
import time
from newspaper import Article
from newspaper.article import ArticleException
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("THENEWSAPI_KEY")


def fetch_articles(query, language, total_articles, page_size, topic_label, published_after, published_before=None):
    all_articles = []
    collected = 0
    page = 1

    print(f"🔍 Fetching articles for: '{query}'")

    while collected < total_articles:
        url = 'https://api.thenewsapi.com/v1/news/all'
        params = {
            'api_token': API_KEY,
            'search': query,
            'language': language,
            'limit': page_size,
            'page': page,
            'published_after': published_after,
            'sort': 'relevance'
        }

        if published_before:
            params['published_before'] = published_before

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', [])
            if not articles:
                break

            for article in articles:
                full_text = extract_full_text(article.get('url', ''))
                all_articles.append({
                    'article_id': article.get('uuid'),
                    'title': article.get('title', ''),
                    'body': full_text,
                    'source': article.get('source', ''),
                    'published_at': article.get('published_at', ''),
                    'url': article.get('url', '')
                })

            collected += len(articles)
            page += 1
            print(f"  → {collected} collected so far")
        else:
            print(f"❌ Error fetching query '{query}':", response.status_code)
            break

    return all_articles


def extract_full_text(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        time.sleep(0.5)  # Avoid rate limiting
        return a.text
    except ArticleException:
        return ''


def save_articles_to_csv(articles, filename):
    if not articles:
        print("⚠️ No articles were collected.")
        return

    df = pd.DataFrame(articles)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} full-text articles to {output_path}")


if __name__ == "__main__":
    # Customize your input here
    search_query = '2024 Champions League final'
    LANGUAGE = 'en'
    TOTAL_ARTICLES = 50
    PAGE_SIZE = 20
    TOPIC_LABEL = '2024+champions+league+final'
    OUTPUT_FILENAME = '2024-CL.csv'
    PUBLISHED_AFTER = '2024-06-01'  
    PUBLISHED_BEFORE = None         

    articles = fetch_articles(
        search_query,
        LANGUAGE,
        TOTAL_ARTICLES,
        PAGE_SIZE,
        TOPIC_LABEL,
        PUBLISHED_AFTER,
        published_before=PUBLISHED_BEFORE
    )
    save_articles_to_csv(articles, OUTPUT_FILENAME)
