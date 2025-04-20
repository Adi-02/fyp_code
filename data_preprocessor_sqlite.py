import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import sqlite3
import pandas as pd

def load_ticker_data_from_sqlite(ticker, db_name="market_data.db"):
    conn = sqlite3.connect(db_name)
    table_name = ticker.replace('.', '_').replace('-', '_')  
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def manual_split_df(stock_df, test_ratio=0.1):
    train_df_dict = {}
    test_df_dict = {}
    tickers = stock_df['Ticker'].unique()
    
    for ticker in tickers:
        df_ticker = stock_df[stock_df['Ticker'] == ticker]
        split_idx = int(len(df_ticker) * (1 - test_ratio))
        train_df_dict[ticker] = df_ticker.iloc[:split_idx].reset_index(drop=True)
        test_df_dict[ticker] = df_ticker.iloc[split_idx:].reset_index(drop=True)
    
    return train_df_dict, test_df_dict

def create_sequences(stock_df, time_series_cols, sentiment_cols, sequence_length=4, overlap=False):
    stock_df = stock_df.sort_values(by=['Ticker', 'Date'])
    tickers = stock_df['Ticker'].unique()

    stock_time_series_input_dict = {}
    stock_sentiment_dict = {}
    stock_time_series_target_dict = {}
    input_ticker_dates = {}
    target_ticker_dates = {}

    for tick in tickers:
        stock_time_series_input_dict[tick] = []
        stock_sentiment_dict[tick] = []
        stock_time_series_target_dict[tick] = []
        input_ticker_dates[tick] = []
        target_ticker_dates[tick] = []

    step_size = 1 if overlap else sequence_length * 2

    for ticker in tickers:
        df_ticker = stock_df[stock_df['Ticker'] == ticker]
        for i in range(0, len(df_ticker) - 2 * sequence_length + 1, step_size):
            df_subset = df_ticker.iloc[i:i + sequence_length].drop(columns=['Ticker'])
            time_series_subset = df_subset[time_series_cols]
            sentiment_subset = df_subset[sentiment_cols]

            input_time_seq = time_series_subset.to_dict(orient='records')
            input_sent_seq = sentiment_subset.to_dict(orient='records')
            input_text = [list(i.values()) for i in input_sent_seq]
            input_ticker_dates[ticker].append(df_subset["Date"].values)
            input_time_series = [np.array(list(i.values())) for i in input_time_seq]

            target_seq = df_ticker.iloc[i + sequence_length:i + 2 * sequence_length][['Close']].values
            target_ticker_dates[ticker].append(df_ticker.iloc[i + sequence_length:i + 2 * sequence_length]["Date"].values)

            stock_time_series_input_dict[ticker].append(input_time_series)
            stock_sentiment_dict[ticker].append(input_text)
            stock_time_series_target_dict[ticker].append(target_seq)

    return stock_time_series_input_dict, stock_sentiment_dict, stock_time_series_target_dict, input_ticker_dates, target_ticker_dates

class DatasetNormalizer:
    def __init__(self):
        self.normalization_stats = {}

    def fit(self, train_input_dict, train_target_dict, time_series_cols):
        for ticker in train_input_dict.keys():
            train_input_arr = np.array(train_input_dict[ticker]) 
            flattened_features = train_input_arr.reshape(-1, len(time_series_cols))
            feature_mean = flattened_features.mean(axis=0)
            feature_std = flattened_features.std(axis=0)

            train_target_arr = np.array(train_target_dict[ticker]).flatten()
            target_mean = train_target_arr.mean()
            target_std = train_target_arr.std()

            self.normalization_stats[ticker] = {
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "target_mean": target_mean,
                "target_std": target_std
            }
        print("Normalization stats computed.")

    def normalize_input(self, input_dict):
        norm_input_dict = {}
        for ticker, sequences in input_dict.items():
            mean = self.normalization_stats[ticker]["feature_mean"]
            std = self.normalization_stats[ticker]["feature_std"]
            norm_sequences = [(np.array(seq) - mean) / std for seq in sequences]
            norm_input_dict[ticker] = norm_sequences
        return norm_input_dict

    def normalize_target(self, target_dict):
        norm_target_dict = {}
        for ticker, targets in target_dict.items():
            mean = self.normalization_stats[ticker]["target_mean"]
            std = self.normalization_stats[ticker]["target_std"]
            norm_targets = [(np.array(seq).flatten() - mean) / std for seq in targets]
            norm_target_dict[ticker] = [target.reshape(-1, 1) for target in norm_targets]
        return norm_target_dict

    def unnormalize_target(self, norm_target_seq, ticker):
        mean = self.normalization_stats[ticker]["target_mean"]
        std = self.normalization_stats[ticker]["target_std"]
        return norm_target_seq * std + mean

class StockSentimentDataset(Dataset):
    def __init__(self, input_data, sentiment_data, target_data):
        self.input_data = [torch.tensor(np.vstack(seq), dtype=torch.float32) for seq in input_data]
        self.sentiment_data = [torch.tensor(np.mean([float(item[1]) for item in seq]), dtype=torch.float32).unsqueeze(0) for seq in sentiment_data]
        self.target_data = [torch.tensor(np.array(seq), dtype=torch.float32) for seq in target_data]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.sentiment_data[idx], self.target_data[idx]

def return_data_sqlite(ticker_to_train, time_series_cols, sentiment_cols, sequence_len, db_name="market_data.db"):
    df = load_ticker_data_from_sqlite(ticker_to_train, db_name)

    df = df.drop_duplicates(subset=['Date'], keep='first')
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'News Titles' in df.columns and 'News Summary' in df.columns and 'Tweets' in df.columns:
        df.drop(columns=["News Titles", "News Summary", "Tweets"], inplace=True)
    df["Close"] = df["Close"].round(2)

    train_df_dict, test_df_dict = manual_split_df(df)

    train_prices = {}
    input_dict_train = {}
    sentiment_dict_train = {}
    target_dict_train = {}
    target_dates_train = {}

    for ticker, df_train in train_df_dict.items():
        input_dict, sentiment_dict, target_dict, _, target_ticker_dates = create_sequences(
            df_train, time_series_cols, sentiment_cols, sequence_length=sequence_len, overlap=False)

        input_dict_train[ticker] = input_dict[ticker]
        sentiment_dict_train[ticker] = sentiment_dict[ticker]
        target_dict_train[ticker] = target_dict[ticker]
        target_dates_train[ticker] = target_ticker_dates[ticker]

        train_prices[ticker] = df_train[['Date', 'Close']].reset_index(drop=True)

    input_dict_test = {}
    sentiment_dict_test = {}
    target_dict_test = {}
    target_dates_test = {}

    for ticker, df_test in test_df_dict.items():
        input_dict, sentiment_dict, target_dict, input_ticker_dates, target_ticker_dates = create_sequences(
            df_test, time_series_cols, sentiment_cols, sequence_length=sequence_len, overlap=True)

        input_dict_test[ticker] = input_dict[ticker]
        sentiment_dict_test[ticker] = sentiment_dict[ticker]
        target_dict_test[ticker] = target_dict[ticker]
        target_dates_test[ticker] = target_ticker_dates[ticker]

    normalizer = DatasetNormalizer()
    normalizer.fit(input_dict_train, target_dict_train, time_series_cols)

    test_input_dict_norm = normalizer.normalize_input(input_dict_test)
    test_target_dict_norm = normalizer.normalize_target(target_dict_test)

    test_dataset = StockSentimentDataset(
        test_input_dict_norm[ticker_to_train],
        sentiment_dict_test[ticker_to_train],
        test_target_dict_norm[ticker_to_train]
    )
    test_dataloader = DataLoader(test_dataset, batch_size=25, shuffle=False)

    return train_prices, target_dates_train, test_dataloader, target_dates_test, normalizer
