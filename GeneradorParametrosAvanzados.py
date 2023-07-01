import pandas as pd
import requests


def anadirParametrosAvanzados(dataframe):
    df = dataframe

    df = anadirRSIConadjclose9(df)
    df = anadirRSIConadjclose14(df)
    df = anadirMACDConadjclose2(df)
    df = anadirMACDConadjclose3(df)
    df = anadirMACDConadjclose4(df)
    df = anadirMACDConadjclose9(df)
    df = anadirMACDsigConadjclose2(df)
    df = anadirMACDsigConadjclose3(df)
    df = anadirMACDsigConadjclose6(df)
    df = anadirMACDsigConadjclose9(df)
    df = anadirMACDhistConadjclose9(df)

    df = anadirRSIConvol(df)

    df = anadirMACDConvol(df)

    df = anadirMACDsigConvol3(df)
    df = anadirMACDsigConvol6(df)
    df = anadirMACDsigConvol9(df)
    # df = anadirMACDhistConvol3(df)
    # df = anadirMACDhistConvol6(df)
    df = anadirMACDhistConvol9(df)
    df = anadirMACDhistConvol12(df)

    df = anadiradjcloselag1(df)
    df = anadiradjcloselag2(df)
    df = anadiradjcloselag3(df)

    df = anadirvolumelag(df)

    # No se añade porque hace 1 query por cada empresa, y realmente no aporta mejora a la precisión
    # df = anadirFearAndGreed(df)

    df = anadirEMAConadjclose20(df)
    df = anadirEMAConadjclose50(df)
    df = anadirEMAConvol20(df)
    df = anadirEMAConvol50(df)

    df = anadirSMAConadjclose20(df)
    df = anadirSMAConadjclose50(df)
    df = anadirSMAConvol5(df)
    df = anadirSMAConvol10(df)
    df = anadirSMAConvol20(df)
    df = anadirSMAConvol50(df)

    df = anadirHammerRangos(df)

    return df


def anadirRSIConadjclose9(dataframe):
    df = dataframe
    df['adjcloseRSI9'] = computeRSI(dataframe['adjclose'], 9)
    return df


def anadirRSIConadjclose14(dataframe):
    df = dataframe
    df['adjcloseRSI14'] = computeRSI(dataframe['adjclose'], 14)
    return df


def anadirMACDConadjclose2(dataframe):
    df = dataframe
    # FastEMA = 12 period EMA from closing price
    # SlowEMA = 26 period EMA from closing price
    df['adjcloseMACD2'] = computeMACD(dataframe['adjclose'], 12, 26, 2)
    return df


def anadirMACDConadjclose3(dataframe):
    df = dataframe
    # FastEMA = 12 period EMA from closing price
    # SlowEMA = 26 period EMA from closing price
    df['adjcloseMACD3'] = computeMACD(dataframe['adjclose'], 12, 26, 3)
    return df


def anadirMACDConadjclose4(dataframe):
    df = dataframe
    # FastEMA = 12 period EMA from closing price
    # SlowEMA = 26 period EMA from closing price
    df['adjcloseMACD4'] = computeMACD(dataframe['adjclose'], 12, 26, 4)
    return df


def anadirMACDConadjclose9(dataframe):
    df = dataframe
    # FastEMA = 12 period EMA from closing price
    # SlowEMA = 26 period EMA from closing price
    df['adjcloseMACD9'] = computeMACD(dataframe['adjclose'], 12, 26, 9)
    return df


def anadirMACDsigConadjclose2(dataframe):
    df = dataframe
    df['adjcloseMACDsig2'] = computeMACDsig(dataframe['adjclose'], 12, 26, 2)
    return df


def anadirMACDsigConadjclose3(dataframe):
    df = dataframe
    df['adjcloseMACDsig3'] = computeMACDsig(dataframe['adjclose'], 12, 26, 3)
    return df


def anadirMACDsigConadjclose6(dataframe):
    df = dataframe
    df['adjcloseMACDsig6'] = computeMACDsig(dataframe['adjclose'], 12, 26, 6)
    return df


def anadirMACDsigConadjclose9(dataframe):
    df = dataframe
    df['adjcloseMACDsig9'] = computeMACDsig(dataframe['adjclose'], 12, 26, 9)
    return df


def anadirMACDhistConadjclose9(dataframe):
    df = dataframe
    df['adjcloseMACDhist9'] = computeMACDhist(dataframe['adjclose'], 12, 26, 9)
    return df


def anadirRSIConvol(dataframe):
    df = dataframe
    periodos = [9, 14, 17]
    for periodo_i in periodos:
        nombreFeature = "volumeRSI" + str(periodo_i)
        df[nombreFeature] = computeRSI(dataframe['volume'], periodo_i)
    return df


def anadirMACDConvol(dataframe):
    df = dataframe
    periodos = [2, 3, 6, 9, 12]
    for periodo_i in periodos:
        nombreFeature = "volumeMACD" + str(periodo_i)
        # FastEMA = 12 period EMA from closing price
        # SlowEMA = 26 period EMA from closing price
        df[nombreFeature] = computeMACD(dataframe['volume'], 12, 26, periodo_i)

    return df


def anadirMACDsigConvol3(dataframe):
    df = dataframe
    df['volumeMACDsig3'] = computeMACDsig(dataframe['volume'], 12, 26, 3)
    return df


def anadirMACDsigConvol6(dataframe):
    df = dataframe
    df['volumeMACDsig6'] = computeMACDsig(dataframe['volume'], 12, 26, 6)
    return df


def anadirMACDsigConvol9(dataframe):
    df = dataframe
    df['volumeMACDsig9'] = computeMACDsig(dataframe['volume'], 12, 26, 9)
    return df


def anadirMACDhistConvol3(dataframe):
    df = dataframe
    df['volumeMACDhist3'] = computeMACDhist(dataframe['volume'], 12, 26, 3)
    return df


def anadirMACDhistConvol6(dataframe):
    df = dataframe
    df['volumeMACDhist6'] = computeMACDhist(dataframe['volume'], 12, 26, 6)
    return df


def anadirMACDhistConvol9(dataframe):
    df = dataframe
    df['volumeMACDhist9'] = computeMACDhist(dataframe['volume'], 12, 26, 9)
    return df


def anadirMACDhistConvol12(dataframe):
    df = dataframe
    df['volumeMACDhist12'] = computeMACDhist(dataframe['volume'], 12, 26, 12)
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


def anadirvolumelag(dataframe):
    df = dataframe
    lag = [1, 2, 3, 4, 5]
    for lag_i in lag:
        nombreFeature = "volumelag" + str(lag_i)
        df[nombreFeature] = computelag(dataframe['volume'], lag_i)
    return df


def anadirFearAndGreed(dataframe):
    df = dataframe
    df['fearandgeed'] = computeFearAndGreed(dataframe)
    return df


def anadirEMAConadjclose20(dataframe):
    df = dataframe
    df['emaadjclose20'] = calculate_ema(dataframe['adjclose'], 20)
    return df


def anadirEMAConadjclose50(dataframe):
    df = dataframe
    df['emaadjclose50'] = calculate_ema(dataframe['adjclose'], 50)
    return df


def anadirEMAConvol20(dataframe):
    df = dataframe
    df['emavolume20'] = calculate_ema(dataframe['volume'], 20)
    return df


def anadirEMAConvol50(dataframe):
    df = dataframe
    df['emavolume50'] = calculate_ema(dataframe['volume'], 50)
    return df


def anadirSMAConadjclose20(dataframe):
    df = dataframe
    df['smaadjclose20'] = calculate_sma(dataframe['adjclose'], 20)
    return df


def anadirSMAConadjclose50(dataframe):
    df = dataframe
    df['smaadjclose50'] = calculate_sma(dataframe['adjclose'], 50)
    return df


def anadirSMAConvol5(dataframe):
    df = dataframe
    df['smavolume5'] = calculate_sma(dataframe['volume'], 5)
    return df


def anadirSMAConvol10(dataframe):
    df = dataframe
    df['smavolume10'] = calculate_sma(dataframe['volume'], 10)
    return df


def anadirSMAConvol20(dataframe):
    df = dataframe
    df['smavolume20'] = calculate_sma(dataframe['volume'], 20)
    return df


def anadirSMAConvol50(dataframe):
    df = dataframe
    df['smavolume50'] = calculate_sma(dataframe['volume'], 50)
    return df


def anadirHammerRangos(dataframe):
    df = dataframe
    # Se generan varias features, iterando con varias combinaciones de parámetros hammer
    diasPreviosA = [1, 2]
    diasPreviosB = [1, 2]
    parametroA = ['high', 'low']
    parametroB = ['low', 'high', 'open']
    parametroC = ['open', 'high']

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
    # Se calculará en porcentaje la fuerzo del patrón martillo, según:
    # Hammer = 100 * caída inicial (valor positivo si cae) * subida final (valor positivo si sube)
    # La caída inicial será: (parametroB  - parametroA), ambos los días previos indicados como parámetro
    # La subida final será: (parametroC - parametroB), donde low será el día previo indicado como parámetro, y el adjclose será de hoy
    caidaInicial = (data[parametroB].shift(diasPreviosB) - data[parametroA].shift(diasPreviosA)) / data[
        parametroA].shift(
        diasPreviosA)
    subidaFinal = (data[parametroC] - data[parametroB].shift(diasPreviosA)) / data[parametroC]
    hammer = 100 * caidaInicial * subidaFinal
    return hammer
