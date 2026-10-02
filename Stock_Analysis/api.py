import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = "1IZH6TJDHW0A1BBR"
URL = "https://www.alphavantage.co/query"


def get_stock_data(symbol: str) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": API_KEY
    }

    response = requests.get(URL, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

   
    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    if "Note" in data:
        raise ValueError(data["Note"])
    if "Information" in data:
        raise ValueError(data["Information"])
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Unexpected API response: {data}")

    
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")


    df.columns = ["open", "high", "low", "close", "volume"]

  
    df = df.astype(float)


    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df
