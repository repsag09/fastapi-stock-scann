import pandas as pd
import numpy as np
from app.screener import apply_template, load_template


def make_uptrend(days=300):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    price = np.linspace(10, 50, num=days) + np.random.normal(0, 0.1, size=days)
    vol = np.random.randint(100000, 200000, size=days)
    df = pd.DataFrame({"Close": price, "Volume": vol}, index=dates)
    df["Open"] = df["Close"] * 0.99
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.98
    return df


def test_minervini_pass():
    template = load_template("minervini")
    # disable RS requirement for unit test environment
    template["params"]["min_rs_pct"] = -999
    df = make_uptrend()
    matched, flags = apply_template(df, template)
    assert matched is True
