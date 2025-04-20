from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas as pd
import json

def get_macro_indicators(date_str, unrate, cpi, vix, yield_spread):
    date = pd.to_datetime(date_str)

    def get_latest(series):
        series = series.dropna()
        try:
            return float(series[:date].iloc[-1])
        except IndexError:
            return None

    def get_exact(series):
        try:
            return float(series.loc[date])
        except KeyError:
            return None

    return pd.DataFrame([{
        'Date': date_str,
        'UNRATE': get_latest(unrate),
        'CPIAUCSL': get_latest(cpi),
        'VIXCLS': get_exact(vix),
        'T10Y2Y': get_exact(yield_spread)
    }])

def get_info():
    load_dotenv()
    fred = Fred(api_key=os.getenv("fred_api_key"))

    unrate = fred.get_series('UNRATE')
    cpi = fred.get_series('CPIAUCSL')
    vix = fred.get_series('VIXCLS')
    yield_spread = fred.get_series('T10Y2Y')

    return unrate, cpi, vix, yield_spread

if __name__ == "__main__":
    unrate, cpi, vix, yield_spread = get_info()
    dates = pd.bdate_range("2025-03-04", "2025-03-24")

    results = []
    for date in dates:
        formatted_date = date.strftime("%Y-%m-%d")
        result = get_macro_indicators(formatted_date, unrate, cpi, vix, yield_spread)
        results.append(result)

    with open("financial_data/macro_data.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Data written to macro_data.json")
