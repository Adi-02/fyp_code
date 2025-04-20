import alpha_vantage
import wsj_scraper
import twitter_scraper 
import rag 
import db_manager
import macro_economic_indicators
import pandas as pd
from datetime import datetime
import time

def update_data_for_ticker(ticker, date, include=False):
    financial_data, news = alpha_vantage.get_data_for_dates(ticker, date, date)
    news = news[:5]
    twitter = twitter_scraper.get_twitter_text_for_day(ticker, date)[:5]
    if include:
        general_news = wsj_scraper.get_general_news(date)[:5]
        news_text = {"news" : news, "general_news" : general_news, "twitter" : twitter}
    else:
        news_text = {"news" : news, "general_news" : None, "twitter" : twitter}
    unrate, cpi, vix, yield_spread = macro_economic_indicators.get_info()
    macro_data = macro_economic_indicators.get_macro_indicators(date, unrate, cpi, vix, yield_spread)
    financial_data["Date"] = pd.to_datetime(financial_data["Date"])
    macro_data["Date"] = pd.to_datetime(macro_data["Date"])
    merged_df = pd.merge(financial_data, macro_data, how="left", on="Date")
    manager = rag.VectorStoreManager()
    news_sentiment_score = rag.run_rag_pipeline_for_ticker_to_generate_score(manager, ticker, date, news_text)
    merged_df["news_sentiment_score"] = news_sentiment_score
    merged_df["ticker"] = ticker
    merged_df["Date"] = date
    if merged_df.empty:
        print("Data has not been recieved from alpha vantage yet")
    else:
        db_manager.update_columns_in_db(ticker, date, merged_df)
    

if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "BA", "DIS", "F", "GOOG", "KO", "MO", "MSFT", "T"]
    dates = pd.bdate_range("2025-03-04", "2025-03-31")
    for ticker in tickers:
        time.sleep(10)
        for date in dates:
            formatted_date = datetime.strftime(date, "%Y-%m-%d")
            print(f"Processing {ticker}: {formatted_date}")
            update_data_for_ticker(ticker, formatted_date)
            time.sleep(10)
    

