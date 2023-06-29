import pandas as pd

def anadirParametrosAvanzados(dataframe):
    df=dataframe
    df=anadirRSIConadjclose14(df)
    df=anadirMACDConadjclose9(df)
    df = anadirMACDConadjclose9(df)
    df = anadirMACDsigConadjclose9(df)
    df = anadirMACDhistConadjclose9(df)

    df=anadirRSIConvol14(df)
    df=anadirMACDConvol9(df)
    df = anadirMACDConvol9(df)
    df = anadirMACDsigConvol9(df)
    df = anadirMACDhistConvol9(df)

    df = anadiradjcloselag1(df)
    df = anadiradjcloselag2(df)
    df = anadiradjcloselag3(df)

    df = anadirvolumelag1(df)
    df = anadirvolumelag2(df)
    df = anadirvolumelag3(df)
    return df

def anadirRSIConadjclose14(dataframe):
    df=dataframe
    df['adjcloseRSI14'] = computeRSI(dataframe['adjclose'], 14)
    return df

def anadirMACDConadjclose9(dataframe):
    df=dataframe
    # FastEMA = 12 period EMA from closing price
    # SlowEMA = 26 period EMA from closing price
    df['adjcloseMACD9'] = computeMACD(dataframe['adjclose'], 12, 26, 9)
    return df

def anadirMACDsigConadjclose9(dataframe):
    df=dataframe
    df['adjcloseMACDsig9'] = computeMACDsig(dataframe['adjclose'], 12, 26, 9)
    return df

def anadirMACDhistConadjclose9(dataframe):
    df=dataframe
    df['adjcloseMACDhist9'] = computeMACDhist(dataframe['adjclose'], 12, 26, 9)
    return df


def anadirRSIConvol14(dataframe):
    df=dataframe
    df['volumeRSI14'] = computeRSI(dataframe['volume'], 14)
    return df

def anadirMACDConvol9(dataframe):
    df=dataframe
    # FastEMA = 12 period EMA from closing price
    # SlowEMA = 26 period EMA from closing price
    df['volumeMACD9'] = computeMACD(dataframe['volume'], 12, 26, 9)
    return df

def anadirMACDsigConvol9(dataframe):
    df=dataframe
    df['volumeMACDsig9'] = computeMACDsig(dataframe['volume'], 12, 26, 9)
    return df

def anadirMACDhistConvol9(dataframe):
    df=dataframe
    df['volumeMACDhist9'] = computeMACDhist(dataframe['volume'], 12, 26, 9)
    return df

def anadiradjcloselag1(dataframe):
    df = dataframe
    df['adjcloselag1'] = computelag(dataframe['adjclose'], 1)
    return df

def anadiradjcloselag2(dataframe):
    df = dataframe
    df['adjcloselag2'] = computelag(dataframe['adjclose'], 2)
    return df

def anadiradjcloselag3(dataframe):
    df = dataframe
    df['adjcloselag3'] = computelag(dataframe['adjclose'], 3)
    return df


def anadirvolumelag1(dataframe):
    df = dataframe
    df['volumelag1'] = computelag(dataframe['volume'], 1)
    return df

def anadirvolumelag2(dataframe):
    df = dataframe
    df['volumelag2'] = computelag(dataframe['volume'], 2)
    return df

def anadirvolumelag3(dataframe):
    df = dataframe
    df['volumelag3'] = computelag(dataframe['volume'], 3)
    return df

# Calculadora de RSI
def computeRSI(data, time_window):
    diff = data.diff(1).dropna()  # diff in one field(one day)

    # this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[diff > 0]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[diff < 0]

    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg = up_chg.ewm(com=time_window - 1, min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1, min_periods=time_window).mean()

    rs = abs(up_chg_avg / down_chg_avg)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# Calculadora de MACD
def computeMACD(data, n_fast, n_slow, n_smooth):
    fastEMA = data.ewm(span=n_fast, min_periods=n_slow).mean()
    slowEMA = data.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = pd.Series(fastEMA - slowEMA, name='MACD')
    return MACD

# Calculadora de MACDsig
def computeMACDsig(data, n_fast, n_slow, n_smooth):
    fastEMA = data.ewm(span=n_fast, min_periods=n_slow).mean()
    slowEMA = data.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = pd.Series(fastEMA - slowEMA, name='MACD')
    MACDsig = pd.Series(MACD.ewm(span=n_smooth, min_periods=n_smooth).mean(), name='MACDsig')
    return MACDsig

# Calculadora de MACDhist
def computeMACDhist(data, n_fast, n_slow, n_smooth):
    fastEMA = data.ewm(span=n_fast, min_periods=n_slow).mean()
    slowEMA = data.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = pd.Series(fastEMA - slowEMA, name='MACD')
    MACDsig = pd.Series(MACD.ewm(span=n_smooth, min_periods=n_smooth).mean(), name='MACDsig')
    MACDhist = pd.Series(MACD - MACDsig, name='MACDhist')
    return MACDhist

# Calculadora de rentabilidad media leyendo el parámetro INCREMENTO
def computeRentabilidadMediaFromIncremento(data):
    return data.loc[:, 'INCREMENTO'].mean()

def computelag(data, lag):
    # Variación en porcentaje
    return 100*(data-data.shift(lag))/data