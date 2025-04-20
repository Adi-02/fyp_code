import torch 
import torch.nn as nn 



class LSTMAttentionMLP(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, forecast_horizon, mlp_hidden_dim):
        super(LSTMAttentionMLP, self).__init__()
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

        self.attn = nn.Linear(lstm_hidden_dim, 1)  

        self.fc = nn.Linear(lstm_hidden_dim, forecast_horizon)

        self.mlp = nn.Sequential(
            nn.Linear(forecast_horizon + 1, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, forecast_horizon)
        )

    def forward(self, x, sentiment_score):
        lstm_out, _ = self.lstm(x)  

        attn_scores = self.attn(lstm_out)  
        attn_weights = torch.softmax(attn_scores, dim=1)  

        context = torch.sum(attn_weights * lstm_out, dim=1)  
        raw_preds = self.fc(context)  

        combined = torch.cat([raw_preds, sentiment_score], dim=1)

        corrected_preds = self.mlp(combined)

        return corrected_preds
