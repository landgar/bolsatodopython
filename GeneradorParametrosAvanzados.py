import pandas as pd
import requests


def anadirParametrosAvanzados(dataframe):
    df = dataframe

    df = anadirRSI(df)
    df = anadirMACD(df)
    df = anadirMACDsig(df)
    df = anadirMACDhist(df)
    df = anadirlag(df)
    df = anadirFearAndGreed(df)
    df = anadirEMA(df)
    df = anadirSMA(df)
    df = anadirHammerRangos(df)

    return df


def anadirRSI(dataframe):
    df = dataframe
    periodos = [9]
    parametro = ['adjclose']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "RSI" + parametro_i + str(periodo_i)
            # FastEMA = 12 period EMA from closing price
            # SlowEMA = 26 period EMA from closing price
            df[nombreFeature] = computeRSI(dataframe[parametro_i], periodo_i)
    return df


def anadirMACD(dataframe):
    df = dataframe
    periodos = [9]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "MACD" + parametro_i + str(periodo_i)
            # FastEMA = 12 period EMA from closing price
            # SlowEMA = 26 period EMA from closing price
            df[nombreFeature] = computeMACD(dataframe[parametro_i], 12, 26, periodo_i)
    return df


def anadirMACDsig(dataframe):
    df = dataframe
    periodos = [9, 14, 20]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "MACDsig" + parametro_i + str(periodo_i)
            df[nombreFeature] = computeMACDsig(dataframe[parametro_i], 12, 26, periodo_i)
    return df


def anadirMACDhist(dataframe):
    df = dataframe
    periodos = [9, 14]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "MACDhist" + parametro_i + str(periodo_i)
            df[nombreFeature] = computeMACDhist(dataframe[parametro_i], 12, 26, periodo_i)
    return df


def anadirlag(dataframe):
    df = dataframe
    lag = [1, 2, 3]
    parametro = ['low', 'high', 'volume']
    for lag_i in lag:
        for parametro_i in parametro:
            nombreFeature = "lag" + parametro_i + str(lag_i)
            df[nombreFeature] = computelag(dataframe[parametro_i], lag_i)
    return df


def anadirFearAndGreed(dataframe):
    df = dataframe
    df['feargreed'] = computeFearAndGreed(dataframe)
    return df


def anadirEMA(dataframe):
    df = dataframe
    periodo = [5, 10]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodo:
        for parametro_i in parametro:
            nombreFeature = "ema" + parametro_i + str(periodo_i)
            df[nombreFeature] = calculate_ema(dataframe[parametro_i], periodo_i)
    return df


def anadirSMA(dataframe):
    df = dataframe
    periodos = [5, 10]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "sma" + parametro_i + str(periodo_i)
            df[nombreFeature] = calculate_sma(dataframe[parametro_i], periodo_i)
    return df


def anadirHammerRangos(dataframe):
    df = dataframe
    # Se generan varias features, iterando con varias combinaciones de parámetros hammer
    # [1, 2, 3, 4, 10]
    # ['adjclose', 'volume', 'close', 'high', 'low', 'open']
    diasPreviosA = [1, 2]
    diasPreviosB = [1, 2]
    parametroA = ['high', 'low']
    parametroB = ['high', 'low']
    parametroC = ['high', 'low']

    for diasPreviosA_i in diasPreviosA:
        for diasPreviosB_i in diasPreviosB:
            for parametroA_i in parametroA:
                for parametroB_i in parametroB:
                    for parametroC_i in parametroC:
                        nombreFeature = "hammer" + str(diasPreviosA_i) + "y" + str(
                            diasPreviosB_i) + parametroA_i + parametroB_i + parametroC_i
                        df[nombreFeature] = calculadoraHammer(data=dataframe, diasPreviosA=diasPreviosA_i,
                                                              diasPreviosB=diasPreviosB_i, parametroA=parametroA_i,
                                                              parametroB=parametroB_i, parametroC=parametroC_i)

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


# Calculadora de rentabilidad mediana leyendo el parámetro INCREMENTO
def computeRentabilidadMediaFromIncremento(data):
    return data['INCREMENTO'].median()


def computelag(data, lag):
    # Variación en porcentaje
    return 100 * (data - data.shift(lag)) / data


# Calculadora de Fear and Greed
def computeFearAndGreed(data):
    # Se obtiene el listado de fear and greed index histórico (varios años) hasta hoy
    r = requests.get('https://api.alternative.me/fng/?limit=0')

    df = pd.DataFrame(r.json()['data'])
    df.value = df.value.astype(int)
    df.timestamp = pd.to_datetime(df.timestamp, unit='s')
    df.set_index(df.timestamp, inplace=True)
    df.rename(columns={'value': 'fear_greed'}, inplace=True)
    df['date'] = df['timestamp'].astype(str).str[:10]

    cols = ['date']
    data2 = data
    data2 = data2.join(df.set_index(cols), on=cols)
    return data2['fear_greed']


# Calculadora de EMA
def calculate_ema(data, days, smoothing=2):
    return data.ewm(span=days, adjust=False).mean()


# Calculadora de SMA
def calculate_sma(data, days):
    return data.rolling(window=days).mean()


# CalculadoraHammer
def calculadoraHammer(data, diasPreviosA, diasPreviosB, parametroA="open", parametroB="low", parametroC="adjclose"):
    # Se calculará en tanto por uno la fuerza del patrón martillo, según:
    # Hammer = caída inicial (valor positivo si cae) * subida final (valor positivo si sube)
    # La caída inicial será: (parametroB  - parametroA), ambos los días previos indicados como parámetro
    # La subida final será: (parametroC - parametroB), donde low será el día previo indicado como parámetro, y el adjclose será de hoy
    caidaInicial = (data[parametroB].shift(diasPreviosB) - data[parametroA].shift(diasPreviosA)) / data[
        parametroA].shift(
        diasPreviosA)
    subidaFinal = (data[parametroC] - data[parametroB].shift(diasPreviosA)) / data[parametroC]
    hammer = caidaInicial * subidaFinal
    return hammer
