import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os


load_dotenv()
alpha_vantage_api_key = os.getenv("alpha_vantage_key")
BASE_URL = "https://www.alphavantage.co/query"

def fetch_intraday_data(symbol, interval, output_size="compact"):
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": alpha_vantage_api_key,
        "outputsize": output_size,
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if f"Time Series ({interval})" in data:
        timeseries = data[f"Time Series ({interval})"]
        df = pd.DataFrame.from_dict(timeseries, orient="index")
        df.columns = ["open", "high", "low", "close", "volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        print("Error fetching data:", data)
        return pd.DataFrame()

def fetch_daily_data(symbol, start_date=None, end_date=None, output_size="compact"):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": alpha_vantage_api_key,
        "outputsize": output_size,
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        timeseries = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(timeseries, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date', "4. close" : "Close", "5. volume" : "Volume"}, inplace=True)
        
        df = df[["Date", "Close", "Volume"]]
        for col in df.columns:
            if col != "Date": 
                try:
                    df[col] = pd.to_numeric(df[col]) 
                except Exception:
                    raise "Error has occured"
                
        if start_date!= None and end_date != None:
            df["Date"] = pd.to_datetime(df["Date"])
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            mask = (df["Date"] >= start) & (df["Date"] <= end)
            filtered_df = df.loc[mask]
            filtered_df = filtered_df.sort_index()
            return filtered_df

        df = df.sort_index()
        return df
    else:
        print("Error fetching data:", data)
        return pd.DataFrame()

def fetch_technical_indicator(symbol, indicator, interval="daily", time_period=60):
    params = {
        "function": indicator,
        "symbol": symbol,
        "interval": interval,
        "time_period": time_period,
        "series_type": "close",
        "apikey": alpha_vantage_api_key,
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    technical_key = f"Technical Analysis: {indicator}"
    if technical_key in data:
        timeseries = data[technical_key]
        df = pd.DataFrame.from_dict(timeseries, orient='index')

        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)

        for col in df.columns:
            if col != "Date": 
                try:
                    df[col] = pd.to_numeric(df[col]) 
                except Exception:
                    raise "Error has occured"
        return df
    else:
        print("Error fetching data:", data)
        return pd.DataFrame()


def fetch_statement(symbol, function_type):
    params = {
        "function": function_type,
        "symbol": symbol,
        "apikey": alpha_vantage_api_key
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "quarterlyReports" in data:
        quaterly_reports = data["quarterlyReports"]
        df = pd.DataFrame(quaterly_reports)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.rename(columns={"fiscalDateEnding" : "Date"})
        return df
    
    elif "quarterlyEarnings" in data:
        quaterly_reports = data["quarterlyEarnings"]
        df = pd.DataFrame(quaterly_reports)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.rename(columns={"fiscalDateEnding" : "Date"})
        return df
    
    else:
        print("Error fetching data:", data)
        return pd.DataFrame()

def fetch_market_news_sentiment(tickers=None, topics=None, time_from=None, time_to=None, sort="LATEST", limit=10):
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": alpha_vantage_api_key,
        "tickers": tickers,
        "topics": topics,
        "time_from": time_from,
        "time_to": time_to,
        "sort": sort,
        "limit": limit,
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "feed" in data:
        news_feed = data["feed"]
        
        processed_feed = []
        for item in news_feed:
            base_data = {
                "title": item.get("title"),
                "url": item.get("url"),
                "time_published": item.get("time_published"),
                "source": item.get("source"),
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "topics": item.get("topics"),  
                "ticker_sentiment": item.get("ticker_sentiment"),  
            }
            processed_feed.append(base_data)
        
        df = pd.DataFrame(processed_feed)
        df = df.drop_duplicates(subset="title")

        df = df.head(limit)

        return df
    else:
        print("Error fetching data:", data)
        return pd.DataFrame()




def get_data_for_dates(ticker, start_date, end_date):
    daily_data_df = fetch_daily_data(ticker, start_date, end_date)

    ema_10_df = fetch_technical_indicator(ticker, "EMA", time_period=10)
    ema_50_df = fetch_technical_indicator(ticker, "EMA", time_period=50)
    ema_200_df = fetch_technical_indicator(ticker, "EMA", time_period=200)
    ema_10_df = ema_10_df.rename(columns={"EMA" : "EMA_10"})
    ema_50_df = ema_50_df.rename(columns={"EMA" : "EMA_50"})
    ema_200_df = ema_200_df.rename(columns={"EMA" : "EMA_200"})


    rsi_df = fetch_technical_indicator(ticker, "RSI")
    macd_df = fetch_technical_indicator(ticker, "MACD")
    macd_df = macd_df[["Date", "MACD_Hist"]]
    bbands_df = fetch_technical_indicator(ticker, "BBANDS")
    bbands_df["BBANDS_Width"] = bbands_df["Real Upper Band"] - bbands_df["Real Lower Band"]
    bbands_df = bbands_df[["Date", "BBANDS_Width"]]
    atr_df = fetch_technical_indicator(ticker, "ATR")
    obv_df = fetch_technical_indicator(ticker, "OBV")

    for df in [daily_data_df, ema_10_df, ema_50_df, ema_200_df, rsi_df, macd_df, bbands_df, atr_df, obv_df]:
        df["Date"] = pd.to_datetime(df["Date"])

    merged_df = daily_data_df.merge(ema_10_df, on="Date", how="left")
    merged_df = merged_df.merge(ema_50_df, on="Date", how="left")
    merged_df = merged_df.merge(ema_200_df, on="Date", how="left")
    merged_df = merged_df.merge(rsi_df, on="Date", how="left")
    merged_df = merged_df.merge(macd_df, on="Date", how="left")
    merged_df = merged_df.merge(bbands_df, on="Date", how="left")
    merged_df = merged_df.merge(atr_df, on="Date", how="left")
    merged_df = merged_df.merge(obv_df, on="Date", how="left")
    

    start = pd.to_datetime(start_date).strftime("%Y%m%d")
    end = pd.to_datetime(end_date).strftime("%Y%m%d")
    start += "T0000"
    end += "T2359"
    news = fetch_market_news_sentiment(ticker, time_from=start, time_to=end, limit=10)
    if news.empty:
        print(f"No news data available for {ticker} on {start_date} - {end_date}.")
        return merged_df, []
    news_lst = news["title"].to_list()

    return merged_df, news_lst
    
def news_sentiment_trial():
    unique_news = set()
    for day in range(4, 32):
        time_from = f"202503{day:02d}T0000"
        time_to = f"202503{day:02d}T2359"
        news = fetch_market_news_sentiment("MO", time_from=time_from, time_to=time_to, limit=10)
        if news.empty:
            print(f"No news for {day}")
            continue
        news_titles = news["title"].to_list()
        news_titles = news_titles[:5]
        for title in news_titles:
            if title not in unique_news:
                unique_news.add(title)
                print(f"Day {day}: {title}")
    

if __name__ == "__main__":
    news_sentiment_trial()


