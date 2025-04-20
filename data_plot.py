import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def recursive_evaluate_and_plot(model, test_dataloader, normalizer, test_target_ticker_dates, train_prices, ticker_to_train, device):
    model.eval()

    all_predictions = []
    all_actuals = []

    forecast_horizon = model.forecast_horizon

    with torch.no_grad():
        for input_time_series_data, sentiment_info, target in test_dataloader:

            input_time_series_data = input_time_series_data.to(device)
            target = target.to(device)
            sentiment_info = sentiment_info.to(device)


            preds = []
            current_input = input_time_series_data.clone()

            for step in range(forecast_horizon):
                output = model(current_input, sentiment_info)
                next_day_pred = output[:, 0].unsqueeze(1)
                preds.append(next_day_pred.cpu().numpy())

                next_day_features = current_input[:, -1, :].clone()
                next_day_features[:, 0] = next_day_pred.squeeze(1)
                current_input = torch.cat([current_input[:, 1:, :], next_day_features.unsqueeze(1)], dim=1)

            preds = np.concatenate(preds, axis=1)
            all_predictions.append(preds)
            all_actuals.append(target.cpu().numpy())


    predictions = np.concatenate(all_predictions, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)

    unnorm_predictions = []
    unnorm_actuals = []

    for i in range(len(predictions)):
        pred = predictions[i]
        actual = actuals[i]

        pred_unnorm = normalizer.unnormalize_target(pred, ticker_to_train)
        actual_unnorm = normalizer.unnormalize_target(actual, ticker_to_train)

        unnorm_predictions.append(pred_unnorm)
        unnorm_actuals.append(actual_unnorm)

    unnorm_predictions = np.array(unnorm_predictions)
    unnorm_actuals = np.array(unnorm_actuals)

    from collections import defaultdict
    predictions_per_date = defaultdict(list)
    actuals_per_date = {}

    for seq_idx in range(len(unnorm_predictions)):
        pred_seq = unnorm_predictions[seq_idx]
        actual_seq = unnorm_actuals[seq_idx]
        date_seq = test_target_ticker_dates[ticker_to_train][seq_idx]

        for day_idx in range(len(pred_seq)):
            date = date_seq[day_idx]
            predictions_per_date[date].append(pred_seq[day_idx])
            actuals_per_date[date] = actual_seq[day_idx]

    unique_dates = sorted(predictions_per_date.keys())

    avg_predictions = []
    avg_actuals = []

    for date in unique_dates:
        preds = predictions_per_date[date]
        avg_pred = np.mean(preds)
        avg_predictions.append(avg_pred)

        actual = actuals_per_date[date]
        avg_actuals.append(actual)

    fig, ax = plt.subplots(figsize=(15, 6)) 

    train_dates = train_prices[ticker_to_train]['Date'].values
    train_close = train_prices[ticker_to_train]['Close'].values
    ax.plot(train_dates, train_close, label="Train Close Price", color='green')

    ax.plot(unique_dates, avg_actuals, label="Test Actual Close Price", color='blue')
    ax.plot(unique_dates, avg_predictions, label="Predicted Close Price (Recursive)", color='orange')

    ax.set_title(f"{ticker_to_train} Close Price Prediction (Train + Recursive Test Forecasting)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    fig.autofmt_xdate()
    fig.tight_layout()

    html_str = "<div style='font-family:monospace; font-size: 16px;'>"

    for date, pred in zip(unique_dates[-4:], avg_predictions[-4:]):
        html_str += f"{date} →  Pred: {pred:.2f} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; "

    html_str = html_str.rstrip(" &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; ") + "</div>"

    return fig, html_str