import gradio as gr
import torch
import pandas as pd
from model import LSTMAttentionMLP
from data_plot import recursive_evaluate_and_plot
from db_manager import store_dataframe_per_ticker_to_sqlite
from data_preprocessor_sqlite import return_data_sqlite
import rag
from datetime import datetime
import refresh_current_data

def load_model(ticker, seq_len):
    if seq_len == 4:
        pth = "models/length_4/" + ticker + ".pth"
    elif seq_len == 8:
        pth = "models/length_8/" + ticker + ".pth"  
    else:   
        pth = "models/length_16/" + ticker + ".pth"
    model = LSTMAttentionMLP(input_dim=seq_len, lstm_hidden_dim=128, forecast_horizon=seq_len, mlp_hidden_dim=256)
    if ticker == "AAPL":
        checkpoint = torch.load(pth, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(pth, map_location='cpu'))
    model.eval()
    return model

device = torch.device("cpu")
tickers = ["AAPL", "AMZN", "BA", "DIS", "F", "GOOG", "KO", "MO", "MSFT", "T"]
time_series_cols = ["Close", "EMA_200", "RSI", "MACD_Hist"]
sentiment_cols = ["Date", "news_sentiment_score"]


def predict_and_plot(ticker, sequence_len):
    model = load_model(ticker, sequence_len)
    prices, dates, test_dataloader, target_dates_test, normalizer = return_data_sqlite(
        ticker, time_series_cols, sentiment_cols, sequence_len
    )

    fig, last_4_days_str = recursive_evaluate_and_plot(model, test_dataloader, normalizer, target_dates_test, prices, ticker, device)
    return fig, last_4_days_str

def restore_market_db():
    stock_df = pd.read_csv("Dataset.csv")
    stock_df.drop(columns="Unnamed: 0", inplace=True)
    stock_df = stock_df.drop_duplicates(subset=['Ticker', 'Date'], keep='first')
    stock_df.drop(columns=["News Titles", "News Summary", "Tweets"], inplace=True)
    stock_df["Close"] = stock_df["Close"].round(2)
    store_dataframe_per_ticker_to_sqlite(stock_df)
    return "Market DB restored from Dataset.csv"

def refresh_data():
    for ticker in tickers:
        print("Processing Ticker: ", ticker)
        now = datetime.now()
        print("Current datetime: ", now)

        today_str = now.strftime("%Y-%m-%d")
        print("Date only: ", today_str)

        if now.hour >= 21:
            print("It's after 9 PM. Proceeding to update...")
            if ticker == "AAPL":
                refresh_current_data.update_data_for_ticker(ticker, today_str, include=True)
            else:
                refresh_current_data.update_data_for_ticker(ticker, today_str, include=False)
        else:
            print("It's before 9 PM. Skipping update.")
            return f"""
            <div style="background-color:#fff4e5;padding:10px;border-radius:5px;color:#b15e00;font-weight:bold;">
                It's before 9 PM. Data update is only available after 9 PM.
            </div>
            """
    return f"""<div style="background-color:#e6ffed;padding:10px;border-radius:5px;color:#215732;font-weight:bold;">
                Data updated for all Tickers on {today_str}
            </div>"""


def chat_with_llm(ticker, question):
    manager = rag.VectorStoreManager()
    return rag.rag_conversation_query(manager, ticker, question)


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Price Prediction"):
            gr.Markdown("## Stock Price Prediction Dashboard")
            gr.Markdown("Predicts the next 4 days' close prices based on technical indicators & sentiment scores.")

            with gr.Row():
                ticker_dropdown = gr.Dropdown(choices=tickers, label="Select Ticker", value=tickers[0])
                sequence_dropdown = gr.Dropdown(choices=[4,8,16], label="Select Forecast Length (Days)", value=4)
                predict_button = gr.Button("Predict")
                refresh_button = gr.Button("Refresh DB")
            refresh_alert = gr.HTML("")
            prediction_output = gr.HTML("")
            output_plot = gr.Plot(label="Train, Test & Predicted Prices")
            predict_button.click(fn=predict_and_plot, inputs=[ticker_dropdown, sequence_dropdown], outputs=[output_plot, prediction_output])
            refresh_button.click(fn=refresh_data, outputs=refresh_alert)

        with gr.Tab("Market Chat Assistant"):
            gr.Markdown("## Market Chat Assistant")
            gr.Markdown("Ask questions about recent market sentiment, news, or tweets for a stock.")

            chat_ticker_dropdown = gr.Dropdown(choices=tickers, label="Select Ticker for Chat", value=tickers[0])
            user_input = gr.Textbox(label="Ask your question", placeholder="e.g. What's the market mood for AAPL?")
            chat_button = gr.Button("Ask")
            chat_output = gr.Textbox(label="Answer", lines=10)

            chat_button.click(
                fn=chat_with_llm,
                inputs=[chat_ticker_dropdown, user_input],
                outputs=chat_output
            )

demo.launch()
