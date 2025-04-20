import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()
def get_twitter_text_for_day(ticker, start_date):
    url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    headers = {"X-API-Key": os.getenv("twitter_api_key")}
    params = {"query": f"${ticker} lang:en since:{start_date}"}

    response = requests.request("GET", url, headers=headers, params=params)
    tweets = json.loads(response.text)["tweets"]
    tweets_lst = []
    for i in tweets:
        tweets_lst.append(i["text"])
    return tweets_lst

if __name__ == "__main__":
    twitter_text = get_twitter_text_for_day("AAPL", "2025-03-03", "2025-03-04")
    print(twitter_text)