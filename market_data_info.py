import yfinance as yf
import pandas as pd
from firecrawl import FirecrawlApp
import re
import datetime
import os 
from dotenv import load_dotenv
import json

def get_ema_values(ticker, date):
    start_date = pd.to_datetime(date) - pd.Timedelta(days=250) 
    end_date = pd.to_datetime(date) + pd.Timedelta(days=1)  

    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False)

    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    if date not in df.index.strftime('%Y-%m-%d'):
        print(f"No data found for {date} (likely non-business day)")
        return None

    row = df.loc[df.index.strftime('%Y-%m-%d') == date]

    result = {
        'Close': row['Close'].values[0].item(),
        'EMA_10': row['EMA_10'].values[0],
        'EMA_50': row['EMA_50'].values[0],
        'EMA_200': row['EMA_200'].values[0]
    }
    return result

load_dotenv()
app = FirecrawlApp(api_key=os.getenv("firecrawlkey"))

def extract_financial_metrics(markdown_text, ticker, date):
    metrics = {
        "Ticker": ticker,
        "Date": date
    }

    volume_match = re.search(r'\*\*Volume\*\* \| ([\d\.]+) (Million|Billion)', markdown_text)
    if volume_match:
        volume_value = float(volume_match.group(1))
        volume_unit = volume_match.group(2)
        if volume_unit == "Million":
            metrics['Volume'] = float(volume_value * 1_000_000)
        elif volume_unit == "Billion":
            metrics['Volume'] = float(volume_value * 1_000_000_000)

    rsi_match = re.search(r'\*\*RSI\*\* \| (\d+)', markdown_text)
    if rsi_match:
        metrics['RSI'] = float(rsi_match.group(1))

    macd_match = re.search(r'\*\*MACD\*\* \| ([\d\.\-]+)', markdown_text)
    if macd_match:
        metrics['MACD_Hist'] = float(macd_match.group(1))

    atr_match = re.search(r'\*\*ATR\*\* \| ([\d\.]+)', markdown_text)
    if atr_match:
        metrics['ATR'] = float(atr_match.group(1))

    obv_match = re.search(r'\*\*OBV\*\* \| ([\d\.]+) (Million|Billion)', markdown_text)
    if obv_match:
        obv_value = float(obv_match.group(1))
        obv_unit = obv_match.group(2)
        if obv_unit == "Million":
            metrics['OBV'] = float(obv_value * 1_000_000)
        elif obv_unit == "Billion":
            metrics['OBV'] = float(obv_value * 1_000_000_000)

    return metrics

def scrape_for_business_days(start_date_str="2025-03-03", ticker="AAPL"):
    start_date = start_date_str
    business_days = pd.bdate_range(start=start_date, end=start_date)
    for single_date in business_days:
        formatted_date = single_date.strftime("%Y-%m-%d")
        print(f"Scraping data for: {formatted_date}")
        try:
            url = f"https://aiolux.com/detail/{ticker}/historical?date={formatted_date}"
            response = app.scrape_url(url=url, params={'formats': ['markdown']})
            markdown_data = response.get("markdown", "")

            if markdown_data.strip() == "":
                print(f"No markdown data found for {formatted_date}")
                continue

            return extract_financial_metrics(markdown_data, ticker=ticker, date=formatted_date)

        except Exception as e:
            print(f"Error on {formatted_date}: {e}")

def get_bbands_width_for_date(ticker: str, target_date: str, time_period: int = 60) -> float:
    target_date = pd.to_datetime(target_date)
    start_date = target_date - pd.Timedelta(days=time_period + 30)  
    end_date = target_date + pd.Timedelta(days=1)

    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False)

    data[f'MA{time_period}'] = data['Close'].rolling(window=time_period).mean()
    data[f'STD{time_period}'] = data['Close'].rolling(window=time_period).std()
    data['UpperBB'] = data[f'MA{time_period}'] + (2 * data[f'STD{time_period}'])
    data['LowerBB'] = data[f'MA{time_period}'] - (2 * data[f'STD{time_period}'])
    data['BBANDS_Width'] = data['UpperBB'] - data['LowerBB']

    if target_date in data.index:
        return float(data.loc[target_date, 'BBANDS_Width'].iloc[0])
    else:
        raise ValueError(f"No BBands data available for {target_date.strftime('%Y-%m-%d')}")

def get_data_in_format(ticker_str, date):
    tech_dict = scrape_for_business_days(start_date_str=date, ticker=ticker_str)
    price_ema = get_ema_values(ticker_str, date)
    bbands_width = get_bbands_width_for_date(ticker_str, date)
    bbands_dict = {"BBANDS_Width": bbands_width}
    merged_dict = tech_dict | price_ema
    final_merged_dict = merged_dict | bbands_dict

    return final_merged_dict


if __name__ == "__main__":
    tickers = ["AAPL", "AMZN", "BA", "DIS", "F", "GOOG", "KO", "MO", "MSFT", "T"]
    dates = pd.bdate_range("2025-03-04", "2025-03-24")
    for ticker in tickers:
        ticker_dict = {}
        file_name = "financial_data/" + ticker + "_bbands_width.json"
        for date in dates:
            formatted_date = datetime.datetime.strftime(date, "%Y-%m-%d")
            result = get_bbands_width_for_date(ticker, formatted_date)
            ticker_dict[formatted_date] = result
        with open(file_name, "w") as file:
            json.dump(ticker_dict, file)