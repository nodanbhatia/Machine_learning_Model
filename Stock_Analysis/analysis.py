import pandas as pd


def add_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates simple technical indicators."""
    df = df.copy()
    
   
    df["SMA_20"] = df["close"].rolling(window=20).mean()
    
    
    df["SMA_50"] = df["close"].rolling(window=50).mean()
    
   
    df["Daily Return (%)"] = df["close"].pct_change() * 100

    return df


def get_statistics(df: pd.DataFrame) -> dict:
    """Computes basic statistical metrics from the stock data."""
    latest_close = df["close"].iloc[-1]
    highest_price = df["high"].max()
    lowest_price = df["low"].min()
    avg_volume = df["volume"].mean()
    volatility = df["Daily Return (%)"].std()

    return {
        "Latest Close Price ($)": round(latest_close, 2),
        "Period High ($)": round(highest_price, 2),
        "Period Low ($)": round(lowest_price, 2),
        "Average Daily Volume": f"{int(avg_volume):,}",
        "Volatility (Std Dev %)": round(volatility, 2),
    }
