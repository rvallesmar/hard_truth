import requests
import pandas as pd
import os
from newspaper import Article
from newspaper.article import ArticleException
import time

# ======================
# CONFIGURATION
# ======================
API_KEY = 'pTvo6bnaBeGDyYFWVCoOeo9UAf3HoAwHrIsPqfq5'
search_query = '2021 Olympics sports'
LANGUAGE = 'en'
TOTAL_ARTICLES = 50
PAGE_SIZE = 20
TOPIC_LABEL = 'olympics+2021+basketball'

# Output path
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'olympics_articles.csv')

# ======================
# FETCH ARTICLES
# ======================
all_articles = []
print(f"Fetching articles for: '{search_query}'")
collected = 0
page = 1

while collected < TOTAL_ARTICLES:
    url = 'https://api.thenewsapi.com/v1/news/all'
    params = {
        'api_token': API_KEY,
        'search': search_query,
        'language': LANGUAGE,
        'limit': PAGE_SIZE,
        'page': page,
        'published_after': '2023-01-01',
        'sort': 'relevance'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        articles = data.get('data', [])
        if not articles:
            break

        for article in articles:
            full_text = ''
            url = article.get('url', '')
            try:
                a = Article(url)
                a.download()
                a.parse()
                full_text = a.text
                # sleep to avoid rate-limiting if many articles
                time.sleep(0.5)
            except ArticleException as e:
                full_text = article.get('content') or article.get('description') or ''

            all_articles.append({
                'article_id': article.get('uuid'),
                'title': article.get('title', ''),
                'body': full_text,
                'source': article.get('source', ''),
                'published_at': article.get('published_at', ''),
                'url': url
            })

        collected += len(articles)
        page += 1
        print(f"  → {collected} collected so far")
    else:
        print(f"Error fetching query '{search_query}':", response.status_code)
        break

# ======================
# SAVE TO CSV
# ======================
if all_articles:
    df = pd.DataFrame(all_articles)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved {len(df)} full-text articles to {OUTPUT_CSV_PATH}")
else:
    print("No articles were collected.")
