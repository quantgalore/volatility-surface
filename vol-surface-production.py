# -*- coding: utf-8 -*-
"""
Created in 2024

@author: Quant Galore
"""

from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar
# from self_email import send_message

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy
import mysql.connector
import pytz
import sys

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
engine = sqlalchemy.create_engine('mysql+mysqlconnector://username:password@database-host-name:3306/database-name')
calendar = get_calendar("NYSE")

date = datetime.today().strftime("%Y-%m-%d")

trading_days = ["Monday", "Tuesday", "Wednesday"]

if datetime.today().strftime("%A") not in trading_days:
    sys.exit()

ticker_data = pd.read_sql("weekly_option_tickers", con=engine)
tickers = np.array(ticker_data["tickers"])

term_structure_data_list = []
times = []

for underlying_ticker in tickers:
    
    try:
        
        start_time = datetime.now()
        #
        
        ticker_call_contracts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&as_of={date}&expired=false&limit=1000&apiKey={polygon_api_key}").json()["results"])
        expiration_dates = ticker_call_contracts["expiration_date"].drop_duplicates().values[:2]
        
        atm_option_list = []
        
        for expiration_date in expiration_dates:
            
            term = np.where(expiration_dates==expiration_date)[0][0]
            
            calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/snapshot/options/{underlying_ticker}?expiration_date={expiration_date}&limit=250&contract_type=call&apiKey={polygon_api_key}").json()["results"])
            puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/snapshot/options/{underlying_ticker}?expiration_date={expiration_date}&limit=250&contract_type=put&apiKey={polygon_api_key}").json()["results"])
            
            #####
            
            call_chain = calls.copy()
            call_chain["last_quote.last_updated"] = pd.to_datetime(call_chain["last_quote.last_updated"].values, unit = "ns", utc = True).tz_convert("America/New_York")
            call_chain["distance_from_price"] = abs(call_chain["details.strike_price"] - call_chain["underlying_asset.price"])
            call_chain["days_to_exp"] = round((pd.to_datetime(call_chain["details.expiration_date"])- pd.to_datetime(date)).dt.days)
            call_chain["intrinsic_value"] = call_chain["underlying_asset.price"] - call_chain["details.strike_price"]
            
            atm_call = call_chain[call_chain["intrinsic_value"] < 0].head(1).copy()#[["details.strike_price", "last_quote.bid", "last_quote.ask", "last_quote.midpoint", "implied_volatility", "details.ticker", "days_to_exp"]]
            
            #####
            
            put_chain = puts.copy()
            put_chain["last_quote.last_updated"] = pd.to_datetime(put_chain["last_quote.last_updated"].values, unit = "ns", utc = True).tz_convert("America/New_York")
            put_chain["distance_from_price"] = abs(put_chain["details.strike_price"] - put_chain["underlying_asset.price"])
            put_chain["days_to_exp"] = round((pd.to_datetime(put_chain["details.expiration_date"])- pd.to_datetime(date)).dt.days)
            
            atm_put = put_chain[put_chain["details.strike_price"] == atm_call["details.strike_price"].iloc[0]].head(1).copy()#[["details.strike_price", "last_quote.bid", "last_quote.ask", "last_quote.midpoint", "implied_volatility", "details.ticker", "days_to_exp"]]
            
            atm_option = pd.concat([atm_call.add_prefix("call_"), atm_put.add_prefix("put_")], axis = 1)
            atm_option_list.append(atm_option)
            
            #####
            
        term_structure = pd.concat(atm_option_list)
        
        if len(term_structure) < 2:
            continue
        
        term_structure["strike_vol"] = round((term_structure["call_implied_volatility"] + term_structure["put_implied_volatility"]) / 2*100,2)
        
        slope = round(term_structure["strike_vol"].iloc[1] - term_structure["strike_vol"].iloc[0],2)
        underlying_price = call_chain["underlying_asset.price"].iloc[0]
        straddle_price = round(((term_structure["call_last_quote.ask"].iloc[0] + term_structure["put_last_quote.ask"].iloc[0])),2)
        lower_break_even, upper_break_even = (term_structure["call_details.strike_price"].iloc[0] - straddle_price), (term_structure["call_details.strike_price"].iloc[0] + straddle_price)
        
        minimum_theo_change = round(((upper_break_even - underlying_price) / underlying_price)*100, 2)
        implied_move = round(round((term_structure["strike_vol"].iloc[0] / np.sqrt(252)),2) * np.sqrt(term_structure["call_days_to_exp"].iloc[0]),2)
        
        term_structure_dataframe = pd.DataFrame([{"days_to_exp": term_structure["call_days_to_exp"].iloc[0],
                                                  "atm_strike": (term_structure["call_details.strike_price"].iloc[0] + term_structure["call_details.strike_price"].iloc[1])/2,
                                                  "strike_vol_0": term_structure["strike_vol"].iloc[0],
                                                  "strike_vol_1": term_structure["strike_vol"].iloc[1],
                                                  "slope": slope,
                                                  "ticker": underlying_ticker,
                                                  "minimum_theo_change": minimum_theo_change,
                                                  "implied_move": implied_move,
                                                  "minimum_to_implied": abs(minimum_theo_change - implied_move),
                                                  "call_price": term_structure["call_last_quote.ask"].iloc[0],
                                                  "put_price": term_structure["put_last_quote.ask"].iloc[0],
                                                  "straddle_price": straddle_price,
                                                  "straddle_volume": round(term_structure["call_day.volume"].iloc[0] + term_structure["put_day.volume"].iloc[0]),
                                                  "call_ticker": term_structure["call_details.ticker"].iloc[0],
                                                  "put_ticker": term_structure["put_details.ticker"].iloc[0]}])
        
        term_structure_data_list.append(term_structure_dataframe)
        
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
        print(error_message, underlying_ticker)
        

term_structures = pd.concat(term_structure_data_list).sort_values(by="slope", ascending = True)
term_structures["implied_breakeven_differential"] = term_structures["implied_move"] - term_structures["minimum_theo_change"]
term_structures = term_structures[(term_structures["atm_strike"] >= 10) & (term_structures["implied_breakeven_differential"] > 0.24) & (term_structures["slope"] < 0) & (term_structures["slope"] >= -5) & (term_structures["minimum_theo_change"] < 10) & (term_structures["straddle_volume"] >= 100) & (term_structures["straddle_price"] < 5)].sort_values(by="straddle_price", ascending =True).head(10)
term_structures["date"] = date

subject_string = f"Optimal Straddle Contracts for {pd.to_datetime(date).strftime('%A')}, {date}"

output_list = []

for ticker in term_structures["ticker"]:
    
    ticker_data = term_structures[term_structures["ticker"] == ticker].copy()

    option_string = f"{ticker}: {round(ticker_data['atm_strike'].iloc[0], 2)} straddle has an edge of {round(ticker_data['implied_breakeven_differential'].iloc[0],2)}% for a price of {round(ticker_data['straddle_price'].iloc[0],2)}"
    output_list.append(option_string)

combined_output_string = "\n\n--".join(output_list)
print(combined_output_string)

# send_message(message = combined_output_string, subject = subject_string)

term_structures.to_sql("term_structures", con = engine, if_exists = "replace")


