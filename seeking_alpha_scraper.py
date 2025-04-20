from firecrawl import FirecrawlApp
import pandas as pd
import os 
from dotenv import load_dotenv
import re
import json
import time

load_dotenv()
app = FirecrawlApp(api_key=os.getenv("firecrawlkey"))


def get_news(ticker, start_date):
    pattern = r"### \[([^\]]+)\]"
    print(f"Scraping data for: {start_date}")
    try:
        url = f"https://seekingalpha.com/symbol/{ticker}/news?from={start_date}&to={start_date}"
        response = app.scrape_url(url=url, params={'formats': ['markdown']})
        markdown_data = response.get("markdown", "")

        if markdown_data.strip() == "":
            print(f"No markdown data found for {start_date}")
            return

        titles = re.findall(pattern, markdown_data)
        return titles

    except Exception as e:
        print(f"Error on {start_date}: {e}")



if __name__ == "__main__":
    start_date = "2025-03-04"
    end_date = "2025-03-26"

    business_days = pd.bdate_range(start=start_date, end=end_date)
    tickers = ["AAPL", "AMZN", "BA", "DIS", "F", "GOOG", "KO", "MO", "MSFT", "T"]

    for ticker in tickers:
        print("Processing:", ticker)

        filename = f"financial_data/{ticker}_news.json"

        if os.path.exists(filename):
            with open(filename, "r") as f:
                ticker_data = json.load(f)
        else:
            ticker_data = {}

        os.makedirs("financial_data", exist_ok=True)

        i = 0
        for single_day in business_days:
            if i == 5:
                time.sleep(10)
                i = 0

            date_str = single_day.strftime("%Y-%m-%d")

            if date_str in ticker_data:
                i += 1
                continue

            news_lst = get_news(ticker, date_str)

            if news_lst:
                ticker_data[date_str] = news_lst

                with open(filename, "w") as f:
                    json.dump(ticker_data, f, indent=4)

                print(f"Added news for {ticker} on {date_str}")

            i += 1
