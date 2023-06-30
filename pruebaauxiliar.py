from datetime import datetime
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pylab import rcParams
import warnings

r = requests.get('https://api.alternative.me/fng/?limit=0')

df = pd.DataFrame(r.json()['data'])
df.value = df.value.astype(int)
df.timestamp = pd.to_datetime(df.timestamp, unit='s')
df.set_index(df.timestamp, inplace=True)
df.rename(columns = {'value':'fear_greed'}, inplace=True)

df['date']=df['timestamp'].astype(str).str[:10]

b=1

