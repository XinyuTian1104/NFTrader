
import datetime
import json
import os
import re
import time
import warnings
from datetime import datetime, timedelta
from pprint import pprint

import dotenv
import matplotlib.pyplot as plt
import pandas as pd
import requests
import selenium
import snscrape.modules.twitter as sntwitter
from bs4 import BeautifulSoup
from PIL import Image
from pytrends.request import TrendReq
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from tqdm import tqdm

from utils.config import data_path, header

warnings.filterwarnings('ignore')


def break_period_into_months(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    result = []
    while start <= end:
        # Account for leap year
        last_day = start.replace(day=28) + timedelta(days=4)
        last_day = min(last_day, end)
        result.append((start.strftime('%Y-%m-%d'),
                      last_day.strftime('%Y-%m-%d')))
        start = last_day + timedelta(days=1)
    return result


def fetch_trend(slug, start_date, end_date):
    pytrends = TrendReq()
    keyword = slug
    periods = break_period_into_months(start_date, end_date)
    trend_data_list = []
    for period in periods:
        pytrends.build_payload(
            kw_list=[keyword], timeframe=f"{period[0]} {period[1]}", geo="US")
        # Fetch the data and format it as a Pandas DataFrame
        trend_data = pytrends.interest_over_time().reset_index()
        # trend_data = trend_data.drop("isPartial", axis=1)
        trend_data_list.append(trend_data)
        num_results = len(trend_data)
        print(
            f"Fetched data for {period[0]} to {period[1]} with keyword {keyword}, {num_results} results")
    trend_data_df = pd.concat(trend_data_list)
    return trend_data_df


def fetch_trends():
    raw_data_dir = os.path.join(data_path, 'raw')
    dirs = os.listdir(raw_data_dir)
    for dir_name in dirs:
        dir_path = os.path.join(raw_data_dir, dir_name)
        slug = dir_name
        time_series = pd.read_json(os.path.join(dir_path, 'time_series.json'))
        time_min = time_series['timestamps'].min()
        time_max = time_series['timestamps'].max()
        # convert Tyle "Timestamp" to "yyyy-mm-dd" string
        time_min = time_min.strftime('%Y-%m-%d')
        time_max = time_max.strftime('%Y-%m-%d')

        # write result
        result = fetch_trend(slug, time_min, time_max)
        result.to_csv(os.path.join(dir_path, 'trend.csv'), index=False)

        break


if __name__ == '__main__':
    fetch_trends()
