# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:01:30 2024

@author: quant
"""

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import pytz
import scipy.optimize as optimize
import math
import mysql.connector
import sqlalchemy

from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar
from scipy.stats import norm

def black_scholes(option_type, S, K, t, r, q, sigma):
    """
    Calculate the Black-Scholes option price.
    
    :param option_type: 'call' for call option, 'put' for put option.
    :param S: Current stock price.
    :param K: Strike price.
    :param t: Time to expiration (in years).
    :param r: Risk-free interest rate (annualized).
    :param q: Dividend yield (annualized).
    :param sigma: Stock price volatility (annualized).
    
    :return: Option price.
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    
    if option_type == 'call':
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")

def call_implied_vol(S, K, t, r, option_price):
    q = 0.015
    option_type = "call"
    
    def f_call(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    call_newton_vol = optimize.newton(f_call, x0=0.15, tol=0.05, maxiter=50)
    
    return call_newton_vol
                
def put_implied_vol(S, K, t, r, option_price):
    q = 0.015
    option_type = "put"
    
    def f_put(sigma):
    
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price
    
    put_newton_vol = optimize.newton(f_put, x0=0.15, tol=0.05, maxiter=50)
    
    return put_newton_vol        

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')

tz = pytz.timezone("GMT")

dates = calendar.schedule(start_date = "2023-06-01", end_date = (datetime.today()-timedelta(days=1))).index.strftime("%Y-%m-%d").values

ticker_data = pd.read_sql("weekly_option_tickers", con=engine)

# If you just want a random selection of 100 tickers to save time, keep the below line unchanged
# If you'd like to try ALL tickers, comment out the below line with a # and uncomment the line below it by removing the #
tickers = np.array(ticker_data["tickers"].values)
# tickers = np.array(ticker_data["tickers"].values)

vol_structures = []
terms_out = 2

for date in dates:
    
    times = []
    
    for underlying_ticker in tickers:
        
        try:
            start_time = datetime.now()
            
            underlying = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            underlying.index = pd.to_datetime(underlying.index, unit = "ms", utc = True).tz_convert("America/New_York")
            underlying = underlying[underlying.index.hour < 16].tail(1)
            underlying_price = underlying["c"].iloc[0]
            
            #
            
            ticker_call_contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
            ticker_call_contracts["date"] = pd.to_datetime(ticker_call_contracts["expiration_date"])
            ticker_call_contracts["days_to_exp"] = (ticker_call_contracts["date"] - pd.to_datetime(date)).dt.days
            ticker_call_contracts["distance_from_price"] = abs(ticker_call_contracts["strike_price"] - underlying_price)
            ticker_call_contracts["intrinsic_value"] = underlying_price - ticker_call_contracts["strike_price"]
            
            ticker_put_contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=put&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
            
            expiration_dates = ticker_call_contracts[ticker_call_contracts["days_to_exp"] > 0]["expiration_date"].drop_duplicates().values[:terms_out]
            
            vol_list = []
            
            for expiration_date in expiration_dates:
                
                term = np.where(expiration_dates==expiration_date)[0][0]
                
                atm_option = ticker_call_contracts[(ticker_call_contracts["expiration_date"] == expiration_date) & (ticker_call_contracts["intrinsic_value"] < 0)].head(1)
                atm_put_option = ticker_put_contracts[(ticker_put_contracts["expiration_date"] == expiration_date) & (ticker_put_contracts["strike_price"] ==  atm_option["strike_price"].iloc[0])]
                
                pre_close_timestamp = (pd.to_datetime(date) + timedelta(hours = 15, minutes = 55)).tz_localize("America/New_York").tz_convert(tz).value
                close_timestamp = (pd.to_datetime(date) + timedelta(hours = 16, minutes = 15)).tz_localize("America/New_York").tz_convert(tz).value
                
                try:
                    atm_call = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_option['ticker'].iloc[0]}?timestamp.gte={pre_close_timestamp}&timestamp.lte={close_timestamp}&limit=50000&sort=timestamp&order=desc&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
                    atm_call_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_option['ticker'].iloc[0]}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                    atm_call.index = pd.to_datetime(atm_call.index, origin = "unix").tz_localize(tz).tz_convert("America/New_York")
                    atm_call = atm_call.head(1)
                    atm_call["mid_price"] = (atm_call["bid_price"] + atm_call["ask_price"]) / 2
                    if len(atm_call_ohlcv) >= 1:   
                        atm_call["v"] = atm_call_ohlcv["v"].iloc[0]
                        
                except Exception:
                    atm_call = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_option['ticker'].iloc[0]}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                    atm_call.index = pd.to_datetime(atm_call.index, unit = "ms", utc = True).tz_convert("America/New_York")
                    atm_call = atm_call.tail(1)
                    atm_call["mid_price"] = atm_call["c"]
            
                try:                
                    atm_put = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_put_option['ticker'].iloc[0]}?timestamp.gte={pre_close_timestamp}&timestamp.lte={close_timestamp}&limit=50000&sort=timestamp&order=desc&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
                    atm_put_ohlcv = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_put_option['ticker'].iloc[0]}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                    atm_put.index = pd.to_datetime(atm_put.index, origin = "unix").tz_localize(tz).tz_convert("America/New_York")
                    atm_put = atm_put.head(1)
                    atm_put["mid_price"] = (atm_put["bid_price"] + atm_put["ask_price"]) / 2
                    if len(atm_put_ohlcv) >= 1:   
                        atm_put["v"] = atm_put["v"].iloc[0]
                    
                except Exception:
                    atm_put = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_put_option['ticker'].iloc[0]}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                    atm_put.index = pd.to_datetime(atm_put.index, unit = "ms", utc = True).tz_convert("America/New_York")
                    atm_put = atm_put.tail(1)
                    atm_put["mid_price"] = atm_put["c"]
                
                time_to_expiration = (((atm_option["date"].iloc[0].tz_localize("America/New_York") + timedelta(hours = 16)) - atm_call.index[0]).total_seconds() / 86400) / 252
                
                atm_call_vol = call_implied_vol(S=underlying_price, K=atm_option["strike_price"].iloc[0], t=time_to_expiration, r=.053, option_price=atm_call["mid_price"].iloc[0])
                atm_put_vol = put_implied_vol(S=underlying_price, K=atm_option["strike_price"].iloc[0], t=time_to_expiration, r=.053, option_price=atm_put["mid_price"].iloc[0])
                
                next_day = dates[np.where(dates==date)[0][0]+1]
                
                atm_call_next_day = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_option['ticker'].iloc[0]}/range/1/day/{next_day}/{next_day}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                atm_call_next_day.index = pd.to_datetime(atm_call_next_day.index, unit = "ms", utc = True).tz_convert("America/New_York")
                
                atm_put_next_day = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{atm_put_option['ticker'].iloc[0]}/range/1/day/{next_day}/{next_day}?adjusted=true&sort=asc&limit=1000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
                atm_put_next_day.index = pd.to_datetime(atm_put_next_day.index, unit = "ms", utc = True).tz_convert("America/New_York")
                
                vol_dataset = pd.DataFrame([{f"days_to_expiration_{term}": time_to_expiration *252,
                                            f"atm_call_vol_{term}": atm_call_vol, f"atm_put_vol_{term}": atm_put_vol,
                                            f"atm_strike_{term}": atm_option["strike_price"].iloc[0],
                                            f"strike_vol_{term}": (atm_call_vol + atm_put_vol) / 2,
                                            f"underlying_price_{term}": underlying_price,
                                            f"call_price_{term}": atm_call["mid_price"].iloc[0],
                                            f"call_volume_{term}": atm_call["v"].iloc[0],
                                            f"call_price_next_day_{term}": atm_call_next_day["c"].iloc[0],
                                            f"put_price_{term}": atm_put["mid_price"].iloc[0],
                                            f"put_volume_{term}": atm_put["v"].iloc[0],
                                            f"put_price_next_day_{term}": atm_put_next_day["c"].iloc[0]}])

                
                vol_list.append(vol_dataset)
                
            vol_structure = pd.concat(vol_list, axis=1)
            
            straddle_price = round(vol_structure["call_price_0"].iloc[0] + vol_structure["put_price_0"].iloc[0],2)
            straddle_price_next_day = round(vol_structure["call_price_next_day_0"].iloc[0] + vol_structure["put_price_next_day_0"].iloc[0],2)
            lower_break_even, upper_break_even = (vol_structure["atm_strike_0"].loc[0] - straddle_price), (vol_structure["atm_strike_0"].iloc[0] + straddle_price)
            gross_pnl = round(straddle_price_next_day - straddle_price,2)
            
            minimum_theo_change = round(((upper_break_even - underlying_price) / underlying_price)*100, 2)
            implied_move = round(round((vol_structure["strike_vol_0"].iloc[0]*100 / np.sqrt(252)),2) * np.sqrt(vol_structure["days_to_expiration_0"].iloc[0]),2)
            
            vol_structure["minimum_theo_change"] = minimum_theo_change
            vol_structure["implied_move"] = implied_move
            vol_structure["implied_breakeven_differential"] = vol_structure["implied_move"] - vol_structure["minimum_theo_change"]
            vol_structure["slope"] = round(vol_structure["strike_vol_1"]*100 - vol_structure["strike_vol_0"]*100,2)
            vol_structure["straddle_volume"] = vol_structure["call_volume_0"] + vol_structure["put_volume_0"]
            vol_structure["straddle_price"] = straddle_price
            vol_structure["straddle_price_next_day"] = straddle_price_next_day
            vol_structure["gross_pnl"] = gross_pnl
            vol_structure["ticker"] = underlying_ticker
            vol_structure["date"] = date
            
            vol_structures.append(vol_structure)
            
            end_time = datetime.now()
            seconds_to_complete = (end_time - start_time).total_seconds()
            times.append(seconds_to_complete)
            iteration = round((np.where(tickers==underlying_ticker)[0][0]/len(tickers))*100,2)
            iterations_remaining = len(tickers) - np.where(tickers==underlying_ticker)[0][0]
            average_time_to_complete = np.mean(times)
            estimated_completion_time = (datetime.now() + timedelta(seconds = int(average_time_to_complete*iterations_remaining)))
            time_remaining = estimated_completion_time - datetime.now()
                    
            print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")
            
        except Exception as error_message:
            print(error_message, date, underlying_ticker)
            continue
    
historical_curve = pd.concat(vol_structures)
engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
historical_curve.to_sql("historical_surface", con = engine, if_exists = "replace")

days = historical_curve["date"].drop_duplicates().values
trades = []

for day in days:
    
    day_data_original = historical_curve[historical_curve["date"] == day].copy()
    
    port_size = 10
    
    day_data = day_data_original[(day_data_original["atm_strike_0"] >= 10) & (day_data_original["implied_breakeven_differential"] > 0.24) & (day_data_original["slope"] < 0) & (day_data_original["slope"] >= -5) & (day_data_original["minimum_theo_change"] < 10)  & (day_data_original["straddle_price"] < 5) & (round(day_data_original["days_to_expiration_0"]) > 1) & (round(day_data_original["days_to_expiration_0"]) <= 4) & (day_data_original["straddle_volume"] >= 100)].sort_values(by="straddle_price", ascending=True).head(port_size)
    
    if len(day_data) >= port_size:
        cost = day_data["straddle_price"].sum()
        gross_pnl = day_data["gross_pnl"].sum()
        total_return = round(gross_pnl/cost,2) * 100
        
        trade_data = pd.DataFrame([{"date": day, "cost": cost, "gross_pnl": gross_pnl,"total_return": total_return, "days_to_exp": day_data["days_to_expiration_0"].iloc[0]}])
        # trades.append(day_data)
        trades.append(trade_data)
    
    else:
        continue
    
all_trades = pd.concat(trades)
all_trades["date"] = pd.to_datetime(all_trades["date"])
all_trades = all_trades.set_index("date")
size = 2000
all_trades["cons"] = (size / (all_trades["cost"]*100)).astype(int)
all_trades["gross_pnl"] = all_trades["gross_pnl"] * all_trades["cons"]

all_trades["capital"] = (all_trades["gross_pnl"].cumsum() * 100) + size

plt.figure(dpi=600)
plt.xticks(rotation=45)
plt.xlabel("Date")
plt.ylabel("Capital")
plt.title(f"Growth of ${size}")
plt.plot(all_trades.index, all_trades["capital"], linestyle='-', marker='o', color='skyblue', linewidth=2, markersize=6)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()