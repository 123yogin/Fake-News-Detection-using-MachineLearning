import requests
import pandas as pd
import datetime

API_KEY = '552e6ba0f1f549a9bdcb0912a0023e05'
BASE_URL = 'https://newsapi.org/v2/everything'
QUERY = 'fake news'

def fetch_news(api_key, query):
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=1)  # Fetch news from the last day

    params = {
        'q': query,
        'from': from_date,
        'to': today,
        'apiKey': api_key
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        print(f'Error: {response.status_code}')
        return []

def save_to_csv(articles, filename='new_articles.csv'):
    data = [{'title': article['title'], 'content': article['content'], 'publishedAt': article['publishedAt']}
            for article in articles]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, mode='a', header=not bool(df.empty))  # Append mode

def main():
    articles = fetch_news(API_KEY, QUERY)
    if articles:
        save_to_csv(articles)
        print('News articles fetched and saved successfully.')
    else:
        print('No new articles found.')

if __name__ == '__main__':
    main()
