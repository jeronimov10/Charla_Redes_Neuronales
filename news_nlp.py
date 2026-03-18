from transformers import pipeline
from pathlib import Path
import requests

pipe = pipeline("text-classification", model="ProsusAI/finbert")

API_KEY = Path(__file__).parent.joinpath("API_KEY").read_text().strip()

def news_api():
    keyword = 'Trump'
    date = '2026-03-16'

    url = (
        'https://newsapi.org/v2/everything?'
        f'q={keyword}&'
        f'from={date}&'
        'sortBy=popularity&'
        f'apiKey={API_KEY}'
    )

    response = requests.get(url)
    articles = response.json()['articles']

    filtered = [
        a for a in articles
        if keyword.lower() in (a.get('title') or '').lower()
        or keyword.lower() in (a.get('description') or '').lower()
    ]

    total_score = 0
    num_articles = 0

    for i, article in enumerate(filtered):
        print(f'Title: {article["title"]}')
        print(f'Link: {article["url"]}')
        print(f'Published: {article["description"]}')

        content = article.get('content') or article.get('description') or ''
        if not content:
            continue

        sentimiento = pipe(content)[0]

        print(f'Sentimiento {sentimiento["label"]}, score: {sentimiento["score"]}')
        print('-' * 40)

        if sentimiento['label'] == 'positive':
            total_score += sentimiento['score']
            num_articles += 1
        elif sentimiento['label'] == 'negative':
            total_score -= sentimiento['score']
            num_articles += 1

    if num_articles == 0:
        return 'No hay artículos disponibles'

    final_score = total_score / num_articles
    return f'Sentimiento general: {"Positivo" if final_score >= 0.15 else "Negativo" if final_score <= -0.15 else "Neutral"} {final_score}'

print(news_api())
