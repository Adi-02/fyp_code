from market_data_info import get_data_in_format
from seeking_alpha_scraper import get_news
from twitter_scraper import get_twitter_text_for_day
def get_data_for_date_for_ticker(date, ticker):
    prices_data = get_data_in_format(ticker, date)
    news_lst = get_news(ticker, date)
    return prices_data, news_lst

if __name__ == "__main__":
    prices, news= get_data_for_date_for_ticker("2025-03-26", "AAPL")
    print(prices)
    print(news)

