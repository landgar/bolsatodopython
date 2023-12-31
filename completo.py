# Needed for decrypting
import base64
import datetime
import ftplib
import hashlib
import io
import json
import os
import warnings
from datetime import date
from datetime import timedelta
from random import randint
from time import sleep

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
# Need to install pycryptodome package
# pip install pycryptodome
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from imblearn.combine import *
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

# La siguiente línea se debe ejecutar en Kaggle
# !pip install yfinance

#################################################
#################################################
# PARÁMETROS
#################################################
#################################################

# Para creación de modelo y predicción
carpeta = "/home/t151521/Descargas/prueba/"
descargarInternetParaGenerarModelo = True

# Para creación de modelo
startDate = '01/01/2022'
endDate = '31/12/2022'
cuantasEmpresas = 20
indiceComienzoListaEmpresas = 400
# Para predicción
PREDICCIONcuantasEmpresas = 20
PREDICCIONindiceComienzoListaEmpresas = 1400

# Poner a True si se quiere entrenar y predecir. A False si sólo predecir
ENTRENAR = True

# Para creación de modelo
nombreFicheroCsvBasica = "infosucio.csv"
nombreFicheroCsvAvanzado = "infolimpioavanzadoTarget.csv"
featuresporimportancia = "featuresporimportancia.png"
pathModelo = carpeta + "MODELO.scikit"

# Para predicción
PREDICCIONnombreFicheroCsvBasica = "PREDICCIONinfosucio.csv"
PREDICCIONnombreFicheroCsvAvanzado = "PREDICCIONinfolimpioavanzadoTarget.csv"
PREDICCIONNombreFicheroCSVDondeInvertir = "PREDICCIONdondeinvertir.csv"
# Se toman xx días hacia atrás, hasta ayer (para poder calcular RSI y demás)
PREDICCIONstartDate = date.today() - timedelta(days=200)
PREDICCIONendDate = date.today() - timedelta(days=0)

#################################################
#################################################
# DEFINICIÓN DE FUNCIONES
#################################################
#################################################

try:
    # pip install requests-html
    from requests_html import HTMLSession
except Exception:
    print("""Warning - Certain functionality 
             requires requests_html, which is not installed.

             Install using:
             pip install requests_html

             After installation, you may have to restart your Python session.""")

base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
default_headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}


def build_url(ticker, start_date=None, end_date=None, interval="1d"):
    if end_date is None:
        end_seconds = int(pd.Timestamp("now").timestamp())

    else:
        end_seconds = int(pd.Timestamp(end_date).timestamp())

    if start_date is None:
        start_seconds = 7223400

    else:
        start_seconds = int(pd.Timestamp(start_date).timestamp())

    site = base_url + ticker

    params = {"period1": start_seconds, "period2": end_seconds,
              "interval": interval.lower(), "events": "div,splits"}

    return site, params


def force_float(elt):
    try:
        return float(elt)
    except:
        return elt


def _convert_to_numeric(s):
    if "M" in s:
        s = s.strip("M")
        return force_float(s) * 1_000_000

    if "B" in s:
        s = s.strip("B")
        return force_float(s) * 1_000_000_000

    return force_float(s)


def get_data(ticker, start_date=None, end_date=None, index_as_date=True,
             interval="1d", headers=default_headers):
    '''Downloads historical stock price data into a pandas data frame.  Interval
       must be "1d", "1wk", "1mo", or "1m" for daily, weekly, monthly, or minute data.
       Intraday minute data is limited to 7 days.

       @param: ticker
       @param: start_date = None
       @param: end_date = None
       @param: index_as_date = True
       @param: interval = "1d"
    '''

    if interval not in ("1d", "1wk", "1mo", "1m"):
        raise AssertionError("interval must be of of '1d', '1wk', '1mo', or '1m'")

    # build and connect to URL
    site, params = build_url(ticker, start_date, end_date, interval)
    resp = requests.get(site, params=params, headers=headers)

    if not resp.ok:
        raise AssertionError(resp.json())

    # get JSON response
    data = resp.json()

    # get open / high / low / close data
    frame = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])

    if frame.empty:
        # DEBUG:
        print("De la empresa \"", ticker, "\" no se han podido obtener datos")

    else:
        # get the date info
        temp_time = data["chart"]["result"][0]["timestamp"]

        if interval != "1m":

            # add in adjclose
            frame["adjclose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]
            frame.index = pd.to_datetime(temp_time, unit="s")
            frame.index = frame.index.map(lambda dt: dt.floor("d"))
            frame = frame[["open", "high", "low", "close", "adjclose", "volume"]]

        else:

            frame.index = pd.to_datetime(temp_time, unit="s")
            frame = frame[["open", "high", "low", "close", "volume"]]

        frame['ticker'] = ticker.upper()

        if not index_as_date:
            frame = frame.reset_index()
            frame.rename(columns={"index": "date"}, inplace=True)

    #############################
    # Lo siguiente es sólo en el caso de predecir, no de generar modelo. Es decir, si la fecha final es muy cercana.
    # Además, cuando se entrena se pasa un string. Cuando se predice, un date, con distintos formatos (/ o -).
    # Con esto diferenciamos uno de otro
    import time
    hoy = time.strftime("%Y-%m-%d")  # Formato 2023-08-14
    if isinstance(end_date, str):
        endDateString = end_date
    else:
        endDateString = end_date.strftime("%Y-%m-%d")

    # print("hoy: ", hoy)
    # print("end_date: ", endDateString)

    if hoy == endDateString:
        print("INVENTAMOS EL DÍA DE HOY...")
        ############# INVENTAMOS EL DÍA DE HOY SI ES INCOMPLETO (MERCADO ABIERTO), ya que Yahoo Finance no nos lo dará ################
        # Cuando la fecha de fin coincide con la fecha de hoy, debemos construir el close con el
        # precio actual (minuto más reciente) y lo añadiremos al final del conjunto de datos
        # Import package
        import yfinance as yf

        # Datos por minuto para el último día
        datosPorMinuto = yf.download(tickers=ticker, period="1d", interval="1m")

        if datosPorMinuto.empty:
            # DEBUG:
            print("De la empresa \"", ticker, "\" no se han podido obtener datos")

        else:

            # Se toma el último minuto de hoy (sólo saldrá durante el mercado abierto. Si no, sale lo de ayer)
            # Por tanto, se recomienda ejecutar el programa sólo en mercado abierto, casi al cierre (para que el valor
            # de close que vamos a estimar sea similar al precio de este último minuto)
            datosUltimoMinuto = datosPorMinuto.tail(1)

            # El índice es la fecha, así que se reconvierte como date en una columna más
            datosUltimoMinuto = datosUltimoMinuto.reset_index()
            datosUltimoMinuto.rename(columns={"index": "date"}, inplace=True)

            # Sólo se añadirá tras la apertura del mercado. Si no, no aparecen datos de hoy. Esta condición se validará comprobando si los últimos datos que tenemos tienen la misma fecha que hoy

            ultimaFecha = datosUltimoMinuto['Datetime'].iloc[0]
            ultimaFechaTrozo = ultimaFecha.strftime("%Y-%m-%d")  # Formato 2023-08-14

            DEPURAR = 0
            if DEPURAR == 1 or ultimaFechaTrozo == hoy:
                # Como estamos en mercado abierto, se añadirá los datos de hoy, aunque no estén completos como día finalizado. Por tanto, habrá que asumir el volumen con lo que hay, y la fecha de close como el precio actual
                print(
                    "ATENCIÓN: EL MERCADO ESTÁ ABIERTO, o se ha cerrado y no son todavía las 23:59h, ASÍ QUE INVENTAREMOS LOS DATOS PARA HOY HASTA el último minuto conocido!. Se recomienda la inversión SÓLO cerca del cierre")

                # Para obtener el Open del día, se toma el Open del primer minuto
                openPrimerMinuto = datosPorMinuto['Open'].iloc[0]

                # Para obtener el high del día, se toma el mayor valor hasta ahora
                highMayorDelDia = datosPorMinuto['High'].max()

                # Para obtener el low del día, se toma el menor valor hasta ahora
                lowMenorDelDia = datosPorMinuto['Low'].min()

                # Se toma el volumen acumulado en ese día. Se estimará el volumen completo del día en función de la hora del último minuto
                from datetime import datetime
                # initialize the local time
                horaActual = int(datetime.now().hour)

                factorMultiplicador = 1
                if horaActual == 15:
                    factorMultiplicador = 8
                elif horaActual == 16:
                    factorMultiplicador = 4
                elif horaActual == 17:
                    factorMultiplicador = 2.6
                elif horaActual == 18:
                    factorMultiplicador = 2
                elif horaActual == 19:
                    factorMultiplicador = 1.8
                elif horaActual == 20:
                    factorMultiplicador = 1.3
                elif horaActual == 21:
                    factorMultiplicador = 1.14
                else:
                    factorMultiplicador = 1

                filasRecibidas = len(datosPorMinuto.index)
                # print("filasRecibidas al detalle de minuto: ", filasRecibidas)
                # print("factorMultiplicador: ", factorMultiplicador)
                volumenAcumulado = datosPorMinuto["Volume"].sum() * factorMultiplicador

                # Para obtener el close y el adjclose, se toma el Close del último minuto hasta ahora
                closeDelDia = datosUltimoMinuto['Close'].iloc[0]
                adjCloseDelDia = datosUltimoMinuto['Close'].iloc[0]

                # Se añade la fila creada para hoy, aunque el mercado siga abierto
                filaParaHoyMercadoAbierto = {'date': ultimaFecha.strftime("%Y-%m-%d 00:00:00"),
                                             'open': openPrimerMinuto,
                                             'high': highMayorDelDia,
                                             'low': lowMenorDelDia,
                                             'close': closeDelDia,
                                             'adjclose': adjCloseDelDia,
                                             'volume': volumenAcumulado,
                                             'ticker': ticker.upper()}
                frame = frame.append(filaParaHoyMercadoAbierto, ignore_index=True)

                # print("la fila inventada para hoy es: ")
                # print(filaParaHoyMercadoAbierto)

            else:
                print(
                    "ATENCIÓN: EL MERCADO TODAVÍA NO ESTÁ ABIERTO. Sólo se podrán tomar datos de ayer o antes. NO se recomienda la inversión!!!!!")

            #############################

    return frame


def tickers_sp500(include_company_data=False):
    '''Downloads list of tickers currently listed in the S&P 500 '''
    # get list of all S&P 500 stocks
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-")

    if include_company_data:
        return sp500

    sp_tickers = sp500.Symbol.tolist()
    sp_tickers = sorted(sp_tickers)

    return sp_tickers


def tickers_nasdaq(include_company_data=False):
    '''Downloads list of tickers currently listed in the NASDAQ'''

    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")

    r = io.BytesIO()
    ftp.retrbinary('RETR nasdaqlisted.txt', r.write)

    if include_company_data:
        r.seek(0)
        data = pd.read_csv(r, sep="|")
        return data

    info = r.getvalue().decode()
    splits = info.split("|")

    tickers = [x for x in splits if "\r\n" in x]
    tickers = [x.split("\r\n")[1] for x in tickers if "NASDAQ" not in x != "\r\n"]
    tickers = [ticker for ticker in tickers if "File" not in ticker]

    ftp.close()

    return tickers


def tickers_other(include_company_data=False):
    '''Downloads list of tickers currently listed in the "otherlisted.txt"
       file on "ftp.nasdaqtrader.com" '''
    ftp = ftplib.FTP("ftp.nasdaqtrader.com")
    ftp.login()
    ftp.cwd("SymbolDirectory")

    r = io.BytesIO()
    ftp.retrbinary('RETR otherlisted.txt', r.write)

    if include_company_data:
        r.seek(0)
        data = pd.read_csv(r, sep="|")
        return data

    info = r.getvalue().decode()
    splits = info.split("|")

    tickers = [x for x in splits if "\r\n" in x]
    tickers = [x.split("\r\n")[1] for x in tickers]
    tickers = [ticker for ticker in tickers if "File" not in ticker]

    ftp.close()

    return tickers


def tickers_dow(include_company_data=False):
    '''Downloads list of currently traded tickers on the Dow'''

    site = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"

    table = pd.read_html(site, attrs={"id": "constituents"})[0]

    if include_company_data:
        return table

    dow_tickers = sorted(table['Symbol'].tolist())

    return dow_tickers


def tickers_ibovespa(include_company_data=False):
    '''Downloads list of currently traded tickers on the Ibovespa, Brazil'''

    table = pd.read_html("https://pt.wikipedia.org/wiki/Lista_de_companhias_citadas_no_Ibovespa")[0]
    table.columns = ["Symbol", "Share", "Sector", "Type", "Site"]

    if include_company_data:
        return table

    ibovespa_tickers = sorted(table.Symbol.tolist())

    return ibovespa_tickers


def tickers_nifty50(include_company_data=False, headers={'User-agent': 'Mozilla/5.0'}):
    '''Downloads list of currently traded tickers on the NIFTY 50, India'''

    site = "https://finance.yahoo.com/quote/%5ENSEI/components?p=%5ENSEI"
    table = pd.read_html(requests.get(site, headers=headers).text)[0]

    if include_company_data:
        return table

    nifty50 = sorted(table['Symbol'].tolist())

    return nifty50


def tickers_niftybank():
    ''' Currently traded tickers on the NIFTY BANK, India '''

    niftybank = ['AXISBANK', 'KOTAKBANK', 'HDFCBANK', 'SBIN', 'BANKBARODA', 'INDUSINDBK', 'PNB', 'IDFCFIRSTB',
                 'ICICIBANK', 'RBLBANK', 'FEDERALBNK', 'BANDHANBNK']

    return niftybank


def tickers_ftse100(include_company_data=False):
    '''Downloads a list of the tickers traded on the FTSE 100 index'''

    table = pd.read_html("https://en.wikipedia.org/wiki/FTSE_100_Index", attrs={"id": "constituents"})[0]

    if include_company_data:
        return table

    return sorted(table.EPIC.tolist())


def tickers_ftse250(include_company_data=False):
    '''Downloads a list of the tickers traded on the FTSE 250 index'''

    table = pd.read_html("https://en.wikipedia.org/wiki/FTSE_250_Index", attrs={"id": "constituents"})[0]

    table.columns = ["Company", "Ticker"]

    if include_company_data:
        return table

    return sorted(table.Ticker.tolist())


def get_quote_table(ticker, dict_result=True, headers={'User-agent': 'Mozilla/5.0'}):
    '''Scrapes data elements found on Yahoo Finance's quote page
       of input ticker

       @param: ticker
       @param: dict_result = True
    '''

    site = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker

    tables = pd.read_html(requests.get(site, headers=headers).text)

    data = pd.concat([tables[0], tables[1]])

    data.columns = ["attribute", "value"]

    quote_price = pd.DataFrame(["Quote Price", get_live_price(ticker)]).transpose()
    quote_price.columns = data.columns.copy()

    data = pd.concat([data, quote_price])

    data = data.sort_values("attribute")

    data = data.drop_duplicates().reset_index(drop=True)

    data["value"] = data.value.map(force_float)

    if dict_result:
        result = {key: val for key, val in zip(data.attribute, data.value)}
        return result

    return data


def get_stats(ticker, headers={'User-agent': 'Mozilla/5.0'}):
    '''Scrapes information from the statistics tab on Yahoo Finance
       for an input ticker

       @param: ticker
    '''

    stats_site = "https://finance.yahoo.com/quote/" + ticker + \
                 "/key-statistics?p=" + ticker

    tables = pd.read_html(requests.get(stats_site, headers=headers).text)

    tables = [table for table in tables[1:] if table.shape[1] == 2]

    table = tables[0]
    for elt in tables[1:]:
        table = pd.concat([table, elt])

    table.columns = ["Attribute", "Value"]

    table = table.reset_index(drop=True)

    return table


def get_stats_valuation(ticker, headers={'User-agent': 'Mozilla/5.0'}):
    '''Scrapes Valuation Measures table from the statistics tab on Yahoo Finance
       for an input ticker

       @param: ticker
    '''

    stats_site = "https://finance.yahoo.com/quote/" + ticker + \
                 "/key-statistics?p=" + ticker

    tables = pd.read_html(requests.get(stats_site, headers=headers).text)

    tables = [table for table in tables if "Trailing P/E" in table.iloc[:, 0].tolist()]

    table = tables[0].reset_index(drop=True)

    return table


def _decrypt_yblob_aes(data):
    '''From pydata/pandas-datareader PR#953 - https://github.com/pydata/pandas-datareader/pull/953/commits/ea66d6b981554f9d0262038aef2106dda7138316 '''
    encrypted_stores = data['context']['dispatcher']['stores']
    _cs = data["_cs"]
    _cr = data["_cr"]

    _cr = b"".join(int.to_bytes(i, length=4, byteorder="big", signed=True) for i in json.loads(_cr)["words"])
    password = hashlib.pbkdf2_hmac("sha1", _cs.encode("utf8"), _cr, 1, dklen=32).hex()

    encrypted_stores = base64.b64decode(encrypted_stores)
    assert encrypted_stores[0:8] == b"Salted__"
    salt = encrypted_stores[8:16]
    encrypted_stores = encrypted_stores[16:]

    def EVPKDF(
            password,
            salt,
            keySize=32,
            ivSize=16,
            iterations=1,
            hashAlgorithm="md5",
    ) -> tuple:
        """OpenSSL EVP Key Derivation Function
        Args:
            password (Union[str, bytes, bytearray]): Password to generate key from.
            salt (Union[bytes, bytearray]): Salt to use.
            keySize (int, optional): Output key length in bytes. Defaults to 32.
            ivSize (int, optional): Output Initialization Vector (IV) length in bytes. Defaults to 16.
            iterations (int, optional): Number of iterations to perform. Defaults to 1.
            hashAlgorithm (str, optional): Hash algorithm to use for the KDF. Defaults to 'md5'.
        Returns:
            key, iv: Derived key and Initialization Vector (IV) bytes.
        Taken from: https://gist.github.com/rafiibrahim8/0cd0f8c46896cafef6486cb1a50a16d3
        OpenSSL original code: https://github.com/openssl/openssl/blob/master/crypto/evp/evp_key.c#L78
        """

        assert iterations > 0, "Iterations can not be less than 1."

        if isinstance(password, str):
            password = password.encode("utf-8")

        final_length = keySize + ivSize
        key_iv = b""
        block = None

        while len(key_iv) < final_length:
            hasher = hashlib.new(hashAlgorithm)
            if block:
                hasher.update(block)
            hasher.update(password)
            hasher.update(salt)
            block = hasher.digest()
            for _ in range(1, iterations):
                block = hashlib.new(hashAlgorithm, block).digest()
            key_iv += block

        key, iv = key_iv[:keySize], key_iv[keySize:final_length]
        return key, iv

    key, iv = EVPKDF(password, salt, keySize=32, ivSize=16, iterations=1, hashAlgorithm="md5")

    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    plaintext = cipher.decrypt(encrypted_stores)
    plaintext = unpad(plaintext, 16, style="pkcs7")
    decoded_stores = json.loads(plaintext)
    return decoded_stores


def _parse_json(url, headers={'User-agent': 'Mozilla/5.0'}):
    html = requests.get(url=url, headers=headers).text

    json_str = html.split('root.App.main =')[1].split('(this)')[0].split(';\n}')[0].strip()

    try:
        data = json.loads(json_str)
        # print("type of json_str :", type(data))
        unencrypted_stores = _decrypt_yblob_aes(data)
        json_info = unencrypted_stores['QuoteSummaryStore']
        # print("json_info :", json_info)
    except:
        return '{}'
    # else:
    # return data
    # new_data = json.dumps(data).replace('{}', 'null')
    # new_data = re.sub(r'\{[\'|\"]raw[\'|\"]:(.*?),(.*?)\}', r'\1', new_data)
    # json_info = json.loads(new_data)
    # print("json info :", json_info)
    return json_info


def _parse_table(json_info):
    df = pd.DataFrame(json_info)

    if df.empty:
        return df

    del df["maxAge"]

    df.set_index("endDate", inplace=True)
    df.index = pd.to_datetime(df.index, unit="s")

    df = df.transpose()
    df.index.name = "Breakdown"

    return df


def get_income_statement(ticker, yearly=True):
    '''Scrape income statement from Yahoo Finance for a given ticker

       @param: ticker
    '''

    income_site = "https://finance.yahoo.com/quote/" + ticker + \
                  "/financials?p=" + ticker

    json_info = _parse_json(income_site)
    try:
        if yearly:
            temp = json_info["incomeStatementHistory"]["incomeStatementHistory"]
        else:
            temp = json_info["incomeStatementHistoryQuarterly"]["incomeStatementHistory"]
    except:
        temp = []

    return _parse_table(temp)


def get_balance_sheet(ticker, yearly=True):
    '''Scrapes balance sheet from Yahoo Finance for an input ticker

       @param: ticker
    '''

    balance_sheet_site = "https://finance.yahoo.com/quote/" + ticker + \
                         "/balance-sheet?p=" + ticker

    json_info = _parse_json(balance_sheet_site)

    try:
        if yearly:
            temp = json_info["balanceSheetHistory"]["balanceSheetStatements"]
        else:
            temp = json_info["balanceSheetHistoryQuarterly"]["balanceSheetStatements"]
    except:
        temp = []

    return _parse_table(temp)


def get_cash_flow(ticker, yearly=True):
    '''Scrapes the cash flow statement from Yahoo Finance for an input ticker

       @param: ticker
    '''

    cash_flow_site = "https://finance.yahoo.com/quote/" + \
                     ticker + "/cash-flow?p=" + ticker

    json_info = _parse_json(cash_flow_site)

    if yearly:
        temp = json_info["cashflowStatementHistory"]["cashflowStatements"]
    else:
        temp = json_info["cashflowStatementHistoryQuarterly"]["cashflowStatements"]

    return _parse_table(temp)


def get_financials(ticker, yearly=True, quarterly=True):
    '''Scrapes financials data from Yahoo Finance for an input ticker, including
       balance sheet, cash flow statement, and income statement.  Returns dictionary
       of results.

       @param: ticker
       @param: yearly = True
       @param: quarterly = True
    '''

    if not yearly and not quarterly:
        raise AssertionError("yearly or quarterly must be True")

    financials_site = "https://finance.yahoo.com/quote/" + ticker + \
                      "/financials?p=" + ticker

    json_info = _parse_json(financials_site)

    result = {}

    if yearly:
        temp = json_info["incomeStatementHistory"]["incomeStatementHistory"]
        table = _parse_table(temp)
        result["yearly_income_statement"] = table

        temp = json_info["balanceSheetHistory"]["balanceSheetStatements"]
        table = _parse_table(temp)
        result["yearly_balance_sheet"] = table

        temp = json_info["cashflowStatementHistory"]["cashflowStatements"]
        table = _parse_table(temp)
        result["yearly_cash_flow"] = table

    if quarterly:
        temp = json_info["incomeStatementHistoryQuarterly"]["incomeStatementHistory"]
        table = _parse_table(temp)
        result["quarterly_income_statement"] = table

        temp = json_info["balanceSheetHistoryQuarterly"]["balanceSheetStatements"]
        table = _parse_table(temp)
        result["quarterly_balance_sheet"] = table

        temp = json_info["cashflowStatementHistoryQuarterly"]["cashflowStatements"]
        table = _parse_table(temp)
        result["quarterly_cash_flow"] = table

    return result


def get_holders(ticker, headers={'User-agent': 'Mozilla/5.0'}):
    '''Scrapes the Holders page from Yahoo Finance for an input ticker

       @param: ticker
    '''

    holders_site = "https://finance.yahoo.com/quote/" + \
                   ticker + "/holders?p=" + ticker

    tables = pd.read_html(requests.get(holders_site, headers=headers).text)

    table_names = ["Major Holders", "Direct Holders (Forms 3 and 4)",
                   "Top Institutional Holders", "Top Mutual Fund Holders"]

    table_mapper = {key: val for key, val in zip(table_names, tables)}

    return table_mapper


def get_analysts_info(ticker, headers={'User-agent': 'Mozilla/5.0'}):
    '''Scrapes the Analysts page from Yahoo Finance for an input ticker

       @param: ticker
    '''

    analysts_site = "https://finance.yahoo.com/quote/" + ticker + \
                    "/analysts?p=" + ticker

    tables = pd.read_html(requests.get(analysts_site, headers=headers).text)

    table_names = [table.columns[0] for table in tables]

    table_mapper = {key: val for key, val in zip(table_names, tables)}

    return table_mapper


def get_live_price(ticker):
    '''Gets the live price of input ticker

       @param: ticker
    '''

    df = get_data(ticker, end_date=pd.Timestamp.today() + pd.DateOffset(10))

    return df.close[-1]


def get_live_prices(ticker_list):
    base_quotes_url = 'https://query1.finance.yahoo.com/v7/finance/quote?symbols='
    new_url = base_quotes_url + ','.join(ticker_list)
    resp = requests.get(new_url, headers=default_headers)
    # get JSON response
    data = resp.json()
    results = {result['symbol']: result['regularMarketPrice']
               for result in data['quoteResponse']['result']}
    return results


def _raw_get_daily_info(site):
    session = HTMLSession()

    resp = session.get(site)

    tables = pd.read_html(resp.html.raw_html)

    df = tables[0].copy()

    df.columns = tables[0].columns

    del df["52 Week Range"]

    df["% Change"] = df["% Change"].map(lambda x: float(x.strip("%+").replace(",", "")))

    fields_to_change = [x for x in df.columns.tolist() if "Vol" in x \
                        or x == "Market Cap"]

    for field in fields_to_change:

        if type(df[field][0]) == str:
            df[field] = df[field].map(_convert_to_numeric)

    session.close()

    return df


def get_day_most_active(count: int = 100):
    return _raw_get_daily_info(f"https://finance.yahoo.com/most-active?offset=0&count={count}")


def get_day_gainers(count: int = 100):
    return _raw_get_daily_info(f"https://finance.yahoo.com/gainers?offset=0&count={count}")


def get_day_losers(count: int = 100):
    return _raw_get_daily_info(f"https://finance.yahoo.com/losers?offset=0&count={count}")


def get_top_crypto():
    '''Gets the top 100 Cryptocurrencies by Market Cap'''

    session = HTMLSession()

    resp = session.get("https://finance.yahoo.com/cryptocurrencies?offset=0&count=100")

    tables = pd.read_html(resp.html.raw_html)

    df = tables[0].copy()

    df["% Change"] = df["% Change"].map(lambda x: float(str(x).strip("%"). \
                                                        strip("+"). \
                                                        replace(",", "")))
    del df["52 Week Range"]
    del df["Day Chart"]

    fields_to_change = [x for x in df.columns.tolist() if "Volume" in x \
                        or x == "Market Cap" or x == "Circulating Supply"]

    for field in fields_to_change:

        if type(df[field][0]) == str:
            df[field] = df[field].map(lambda x: _convert_to_numeric(str(x)))

    session.close()

    return df


def get_dividends(ticker, start_date=None, end_date=None, index_as_date=True,
                  headers=default_headers
                  ):
    '''Downloads historical dividend data into a pandas data frame.

       @param: ticker
       @param: start_date = None
       @param: end_date = None
       @param: index_as_date = True
    '''

    # build and connect to URL
    site, params = build_url(ticker, start_date, end_date, "1d")
    resp = requests.get(site, params=params, headers=headers)

    if not resp.ok:
        return pd.DataFrame()

    # get JSON response
    data = resp.json()

    # check if there is data available for dividends
    if "events" not in data["chart"]["result"][0] or "dividends" not in data["chart"]["result"][0]['events']:
        return pd.DataFrame()

    # get the dividend data
    frame = pd.DataFrame(data["chart"]["result"][0]['events']['dividends'])

    frame = frame.transpose()

    frame.index = pd.to_datetime(frame.index, unit="s")
    frame.index = frame.index.map(lambda dt: dt.floor("d"))

    # sort in chronological order
    frame = frame.sort_index()

    frame['ticker'] = ticker.upper()

    # remove old date column
    frame = frame.drop(columns='date')

    frame = frame.rename({'amount': 'dividend'}, axis='columns')

    if not index_as_date:
        frame = frame.reset_index()
        frame.rename(columns={"index": "date"}, inplace=True)

    return frame


def get_splits(ticker, start_date=None, end_date=None, index_as_date=True,
               headers=default_headers
               ):
    '''Downloads historical stock split data into a pandas data frame.

       @param: ticker
       @param: start_date = None
       @param: end_date = None
       @param: index_as_date = True
    '''

    # build and connect to URL
    site, params = build_url(ticker, start_date, end_date, "1d")
    resp = requests.get(site, params=params, headers=headers)

    if not resp.ok:
        raise AssertionError(resp.json())

    # get JSON response
    data = resp.json()

    # check if there is data available for events
    if "events" not in data["chart"]["result"][0]:
        raise AssertionError("There is no data available on stock events, or none have occured")

        # check if there is data available for splits
    if "splits" not in data["chart"]["result"][0]['events']:
        raise AssertionError("There is no data available on stock splits, or none have occured")

    # get the split data
    frame = pd.DataFrame(data["chart"]["result"][0]['events']['splits'])

    frame = frame.transpose()

    frame.index = pd.to_datetime(frame.index, unit="s")
    frame.index = frame.index.map(lambda dt: dt.floor("d"))

    # sort in to chronological order
    frame = frame.sort_index()

    frame['ticker'] = ticker.upper()

    # remove unnecessary columns
    frame = frame.drop(columns=['date', 'denominator', 'numerator'])

    if not index_as_date:
        frame = frame.reset_index()
        frame.rename(columns={"index": "date"}, inplace=True)

    return frame


def get_earnings(ticker):
    '''Scrapes earnings data from Yahoo Finance for an input ticker

       @param: ticker
    '''

    result = {
        "quarterly_results": pd.DataFrame(),
        "yearly_revenue_earnings": pd.DataFrame(),
        "quarterly_revenue_earnings": pd.DataFrame()
    }

    financials_site = "https://finance.yahoo.com/quote/" + ticker + \
                      "/financials?p=" + ticker

    json_info = _parse_json(financials_site)

    if "earnings" not in json_info:
        return result

    temp = json_info["earnings"]

    if temp == None:
        return result

    result["quarterly_results"] = pd.DataFrame.from_dict(temp["earningsChart"]["quarterly"])

    result["yearly_revenue_earnings"] = pd.DataFrame.from_dict(temp["financialsChart"]["yearly"])

    result["quarterly_revenue_earnings"] = pd.DataFrame.from_dict(temp["financialsChart"]["quarterly"])

    return result


### Earnings functions
def _parse_earnings_json(url, headers=default_headers
                         ):
    resp = requests.get(url, headers=headers)

    content = resp.content.decode(encoding='utf-8', errors='strict')

    page_data = [row for row in content.split(
        '\n') if row.startswith('root.App.main = ')][0][:-1]

    page_data = page_data.split('root.App.main = ', 1)[1]

    return json.loads(page_data)


def get_next_earnings_date(ticker):
    base_earnings_url = 'https://finance.yahoo.com/quote'
    new_url = base_earnings_url + "/" + ticker

    parsed_result = _parse_earnings_json(new_url)

    temp = \
        parsed_result['context']['dispatcher']['stores']['QuoteSummaryStore']['calendarEvents']['earnings'][
            'earningsDate'][
            0]['raw']

    return datetime.datetime.fromtimestamp(temp)


def get_earnings_history(ticker):
    '''Inputs: @ticker
       Returns the earnings calendar history of the input ticker with
       EPS actual vs. expected data.'''

    url = 'https://finance.yahoo.com/calendar/earnings?symbol=' + ticker

    result = _parse_earnings_json(url)

    return result["context"]["dispatcher"]["stores"]["ScreenerResultsStore"]["results"]["rows"]


def get_earnings_for_date(date, offset=0, count=1):
    '''Inputs: @date
       Returns a dictionary of stock tickers with earnings expected on the
       input date.  The dictionary contains the expected EPS values for each
       stock if available.'''

    base_earnings_url = 'https://finance.yahoo.com/calendar/earnings'

    if offset >= count:
        return []

    temp = pd.Timestamp(date)
    date = temp.strftime("%Y-%m-%d")

    dated_url = '{0}?day={1}&offset={2}&size={3}'.format(
        base_earnings_url, date, offset, 100)

    result = _parse_earnings_json(dated_url)

    stores = result['context']['dispatcher']['stores']

    earnings_count = stores['ScreenerCriteriaStore']['meta']['total']

    new_offset = offset + 100

    more_earnings = get_earnings_for_date(date, new_offset, earnings_count)

    current_earnings = stores['ScreenerResultsStore']['results']['rows']

    total_earnings = current_earnings + more_earnings

    return total_earnings


def get_earnings_in_date_range(start_date, end_date):
    '''Inputs: @start_date
               @end_date

       Returns the stock tickers with expected EPS data for all dates in the
       input range (inclusive of the start_date and end_date.'''

    earnings_data = []

    days_diff = pd.Timestamp(end_date) - pd.Timestamp(start_date)
    days_diff = days_diff.days

    current_date = pd.Timestamp(start_date)

    dates = [current_date + datetime.timedelta(diff) for diff in range(days_diff + 1)]
    dates = [d.strftime("%Y-%m-%d") for d in dates]

    i = 0
    while i < len(dates):
        try:
            earnings_data += get_earnings_for_date(dates[i])
        except Exception:
            pass

        i += 1

    return earnings_data


def get_currencies(headers={'User-agent': 'Mozilla/5.0'}):
    '''Returns the currencies table from Yahoo Finance'''

    site = "https://finance.yahoo.com/currencies"
    tables = pd.read_html(requests.get(site, headers=headers).text)

    result = tables[0]

    return result


def get_futures(headers={'User-agent': 'Mozilla/5.0'}):
    '''Returns the futures table from Yahoo Finance'''

    site = "https://finance.yahoo.com/commodities"
    tables = pd.read_html(requests.get(site, headers=headers).text)

    result = tables[0]

    return result


def get_undervalued_large_caps(headers={'User-agent': 'Mozilla/5.0'}):
    '''Returns the undervalued large caps table from Yahoo Finance'''

    site = "https://finance.yahoo.com/screener/predefined/undervalued_large_caps?offset=0&count=100"

    tables = pd.read_html(requests.get(site, headers=headers).text)

    result = tables[0]

    return result


def get_quote_data(ticker, headers=default_headers
                   ):
    '''Inputs: @ticker

       Returns a dictionary containing over 70 elements corresponding to the
       input ticker, including company name, book value, moving average data,
       pre-market / post-market price (when applicable), and more.'''

    site = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + ticker

    resp = requests.get(site, headers=default_headers
                        )

    if not resp.ok:
        raise AssertionError("""Invalid response from server.  Check if ticker is
                              valid.""")

    json_result = resp.json()
    info = json_result["quoteResponse"]["result"]

    return info[0]


def get_market_status():
    '''Returns the current state of the market - PRE, POST, OPEN, or CLOSED'''

    quote_data = get_quote_data("^dji")

    return quote_data["marketState"]


def get_premarket_price(ticker):
    '''Inputs: @ticker

       Returns the current pre-market price of the input ticker
       (returns value if pre-market price is available.'''

    quote_data = get_quote_data(ticker)

    if "preMarketPrice" in quote_data:
        return quote_data["preMarketPrice"]

    raise AssertionError("Premarket price not currently available.")


def get_postmarket_price(ticker):
    '''Inputs: @ticker

       Returns the current post-market price of the input ticker
       (returns value if pre-market price is available.'''

    quote_data = get_quote_data(ticker)

    if "postMarketPrice" in quote_data:
        return quote_data["postMarketPrice"]

    raise AssertionError("Postmarket price not currently available.")


# Company Information Functions
def get_company_info(ticker):
    '''Scrape the company information for a ticker

       @param: ticker
    '''
    site = f"https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
    json_info = _parse_json(site)
    json_info = json_info["assetProfile"]
    info_frame = pd.DataFrame.from_dict(json_info,
                                        orient="index",
                                        columns=["Value"])
    info_frame = info_frame.drop("companyOfficers", axis="index")
    info_frame.index.name = "Breakdown"
    return info_frame


def get_company_officers(ticker):
    '''Scrape the company information and return a table of the officers

       @param: ticker
    '''
    site = f"https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
    json_info = _parse_json(site)
    json_info = json_info["assetProfile"]["companyOfficers"]
    info_frame = pd.DataFrame.from_dict(json_info)
    info_frame = info_frame.set_index("name")
    return info_frame


# empresasMaximas: NÚMERO MÁXIMO DE EMPRESAS A LEER. Si quiero todos, poner un número gigante: 999999
def getEmpresas(empresasMaximas, startDate, endDate, offsetEmpresas, mercado):
    empresasMaximasAux = empresasMaximas

    ticker_list = []
    if (mercado == "NASDAQ"):
        # Listado completo de empresas del NASDAQ
        ticker_list = tickers_nasdaq()
    elif (mercado == "SP500"):
        ticker_list = tickers_sp500()
    elif (mercado == "OTHER"):
        ticker_list = tickers_other()
    else:
        raise Exception("MERCADO NO ACEPTADO... NO SE PUEDEN OBTENER EMPRESAS")

    # Se eliminan las N primeras, para poder escoger unas empresas u otras
    ticker_list = ticker_list[offsetEmpresas:]

    # OTRA INFORMACIÓN
    # """ pull historical data for Netflix (NFLX) """
    # nflx = get_data("NFLX")
    # """ get list of all stocks currently in the S&P 500 """
    # sp500_ticker_list = tickers_sp500()
    #
    # """ get other tickers not in NASDAQ (based off nasdaq.com)"""
    # other_tickers = tickers_other()
    #
    # """ get information on stock from quote page """
    # # INFORMACIÓN ESTÁTICA
    # info = get_quote_table("amzn")

    # Se obtienen todos los datos de las empresas
    primero = True
    for i in range(len(ticker_list)):
        empresasMaximasAux = empresasMaximasAux - 1
        if empresasMaximasAux < 0:
            break

        # Esperar aleatoria x segundos
        sleep(randint(1, 4))

        # DEBUG
        print("Empresa " + str(i + 1) + " - Se obtienen los datos de la empresa: ", ticker_list[i])

        try:
            datosEmpresa = get_data(ticker_list[i], start_date=startDate,
                                    end_date=endDate, index_as_date=False)
            if primero:
                primero = False
                # Dataframe vacío
                datoscompletos = pd.DataFrame(datosEmpresa)
            else:
                datoscompletos.reset_index(drop=True, inplace=True)
                datosEmpresa.reset_index(drop=True, inplace=True)
                datoscompletos = pd.concat([datoscompletos, datosEmpresa], axis=0)

        except AssertionError as error:
            print("No se ha podido obtener información de la empresa \"" + ticker_list[
                i] + "\" pero se continúa...")

    return datoscompletos


# Guardar CSV con la información de las empresas
def descargaDatosACsv(cuantasEmpresas, startDate, endDate, carpeta, nombreFicheroCsv, offsetEmpresas):
    mercado = "NASDAQ"
    datos = getEmpresas(cuantasEmpresas, startDate, endDate, offsetEmpresas, mercado)
    guardarDataframeEnCsv(dataframe=datos, filepath=carpeta + nombreFicheroCsv)


def ordenarPorFechaMasReciente(data):
    return data.sort_values(by='date', ascending=False)


def anadirTarget(dataframe, minimoIncrementoEnPorcentaje, periodo):
    df = dataframe
    df['TARGET'] = computeTargetConadjclose(df, minimoIncrementoEnPorcentaje, periodo)
    return df


def anadirIncrementoEnPorcentaje(dataframe, periodo):
    df = dataframe
    df['INCREMENTO'] = computeIncrementoConadjclose(df, periodo)
    return df


def computeTargetConadjclose(data, incrementoEnPorcentaje, periodo):
    df = data
    # Se toma sólo la columna indicada para realizar los cálculos
    df['diff'] = -data.groupby('ticker')['adjclose'].diff(-periodo)  # datos desplazados el periodo

    # Se calcula el target
    target = df.apply(lambda row: gettarget(row, incrementoEnPorcentaje), axis=1)
    return target


def computeIncrementoConadjclose(data, periodo):
    df = data
    # Se toma sólo la columna indicada para realizar los cálculos
    df['diff'] = -data.groupby('ticker')['adjclose'].diff(-periodo)  # datos desplazados el periodo

    # Se calcula el incremento de subida
    incremento = df.apply(lambda row: getPercentage(row['diff'], row['adjclose'] - row['diff']), axis=1)
    return incremento


def gettarget(row, incrementoEnPorcentaje):
    percentage = getPercentage(row['diff'], row['adjclose'] - row['diff'])
    # Se asigna un 1 si se cumple el objetivo, y un 0 si no
    target = getBinary(percentage, incrementoEnPorcentaje)
    return target


def getPercentage(diferencia, base):
    try:
        percentage = (diferencia / base) * 100
    except ZeroDivisionError:
        percentage = float('-inf')
    return percentage


def getBinary(data, umbral):
    binary = data > umbral
    # Para convertir True/False en 1/0
    binary = binary * 1
    return binary


def guardarDataframeEnCsv(dataframe, filepath):
    os.makedirs('folder/subfolder', exist_ok=True)
    dataframe.to_csv(filepath, index=False)


def leerCSV(filepathCSV):
    return pd.read_csv(filepathCSV)


def procesaInformacion(carpetaEntrada, nombreFicheroCsvEntrada, carpetaSalida, nombreFicheroCsvSalida):
    datos = leerCSV(carpetaEntrada + nombreFicheroCsvEntrada)
    # Se cuentan cuántos días de datos hay para cada empresa
    numFilasPorTicker = datos.pivot_table(columns=['ticker'], aggfunc='size')
    minimasFilasExigiblesPorTicker = max(numFilasPorTicker)

    # DEBUG:
    print("Filas por ticker: ", numFilasPorTicker)

    # DEBUG:
    print("Número mínimo de filas por ticket exigibles: ", minimasFilasExigiblesPorTicker)

    # Se eliminan los tickers con poca información (incompletos en días)
    numFilasPorTicker = numFilasPorTicker.to_frame()
    empresas = numFilasPorTicker[numFilasPorTicker[0] >= minimasFilasExigiblesPorTicker]
    datos = datos[datos['ticker'].isin(empresas.index)]

    # Se almacena el fichero limpio básico
    # nombreFicheroCsvSalidaBasico = "infolimpiobasico.csv"
    # guardarDataframeEnCsv(dataframe=datos, guardarIndice=False, filepath=carpetaSalida+nombreFicheroCsvSalidaBasico)

    datos = procesaEmpresas(datos)

    # # Si se añadiera la solución en las propias features, debería salir precisión=100%, y renta muy alta
    # datos['solucion']=datos['TARGET']

    # Se almacena el fichero avanzado con target
    guardarDataframeEnCsv(dataframe=datos, filepath=carpetaSalida + nombreFicheroCsvSalida)


def procesaEmpresa(datos):
    # Se añaden parámetros avanzados
    datos = anadirParametrosAvanzados(dataframe=datos)

    periodo = 3

    print("periodo = " + str(periodo))

    # Se añade el incremento en porcentaje
    datos = anadirIncrementoEnPorcentaje(dataframe=datos, periodo=periodo)

    # Se añade el target
    datos = anadirTarget(dataframe=datos, minimoIncrementoEnPorcentaje=5, periodo=periodo)

    return datos


def procesaEmpresas(datos):
    primero = True
    empresas = datos.groupby("ticker")
    i = 0
    for nombreEmpresa, datosEmpresa in empresas:
        # DEBUG:
        i = i + 1
        print("Procesado de la empresa  " + str(i) + " : ", nombreEmpresa)
        datos = procesaEmpresa(datosEmpresa)
        if primero:
            primero = False
            datoscompletos = pd.DataFrame(datosEmpresa)
        else:

            datoscompletos.reset_index(drop=True, inplace=True)
            datosEmpresa.reset_index(drop=True, inplace=True)
            datoscompletos = pd.concat([datoscompletos, datosEmpresa], axis=0)

    return datoscompletos


def aleatorizarYTrocearEnDos(datos, primeraFraccionTantoPorUno):
    datos_aleatorio = aleatorizarDatos(datos=datos)
    #########################################
    # Se fraccionan los datos de train en: A + B
    fraccion_A = primeraFraccionTantoPorUno
    fraccion_B = 1.00 - fraccion_A

    A = datos_aleatorio.iloc[:int(fraccion_A * len(datos_aleatorio)), :]
    B = datos_aleatorio.iloc[int(fraccion_A * len(datos_aleatorio)):, :]
    return A, B


def aleatorizarDatos(datos):
    return datos.sample(frac=1)


def anadirParametrosAvanzados(dataframe):
    df = dataframe

    df = anadirRSI(df)
    df = anadirMACD(df)
    df = anadirMACDsigydif(df)
    df = anadirMACDhist(df)
    df = anadirlagRelativa(df)
    df = anadirFearAndGreed(df)
    df = anadirEMARelativa(df)
    df = anadirSMARelativa(df)
    df = anadirHammerRangosRelativa(df)
    df = anadirvwapRelativa(df)
    df = anadirDistanciaAbollingerRelativa(df)
    df = anadirATR(df)
    df = anadirCCI(df)
    df = anadirsupernovaTipoA(df)
    df = anadirsupernovaTipoB(df)
    df = anadirsupernovaTipoC(df)
    df = anadirsupernovaTipoD(df)
    df = anadirsupernovaTipoE(df)
    df = anadirsupernovaTipoF(df)
    df = anadiradl(df)
    df = anadirstochastic_oscillator(df)
    df = anadirVolumenRelativo(df)
    df = anadirFeaturesJapanCompetition1(df)
    df = anadirGapAcumulado(df)

    return df


def anadirRSI(dataframe):
    df = dataframe
    periodos = [15, 25, 50]  # SIEMPRE MAYOR O IGUAL QUE 3
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "RSI" + parametro_i + str(periodo_i)
            # FastEMA = 12 period EMA from closing price
            # SlowEMA = 26 period EMA from closing price
            df[nombreFeature] = computeRSI(dataframe[parametro_i], periodo_i)
    return df


def anadirMACD(dataframe):
    df = dataframe
    periodos = [15, 25, 50]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "MACD" + parametro_i + str(periodo_i)
            # FastEMA = 12 period EMA from closing price
            # SlowEMA = 26 period EMA from closing price
            df[nombreFeature] = computeMACD(dataframe[parametro_i], 12, 26, periodo_i)
    return df


def anadirMACDsigydif(dataframe):
    df = dataframe
    periodos = [15, 25, 50]
    parametro = ['adjclose', 'volume']
    lag = [0, 1, 2, 3]
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "MACDsig-" + parametro_i + "-" + str(periodo_i)
            macdsig = computeMACDsig(dataframe[parametro_i], 12, 26, periodo_i)
            df[nombreFeature] = macdsig
            for lag_i in lag:
                dfdesplazado = computelag(dataframe[parametro_i], lag_i)
                macddesplazado = computeMACD(dfdesplazado, 12, 26, periodo_i)
                nombreFeaturedif = "MACDdif-" + parametro_i + "-" + str(periodo_i) + "-" + str(lag_i)
                df[nombreFeaturedif] = macddesplazado - macdsig
    return df


def anadirMACDhist(dataframe):
    df = dataframe
    periodos = [15, 25, 50]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "MACDhist" + parametro_i + str(periodo_i)
            df[nombreFeature] = computeMACDhist(dataframe[parametro_i], 12, 26, periodo_i)
    return df


def anadirlagRelativa(dataframe):
    df = dataframe
    lag = [1, 2, 5, 10, 15]
    parametro = ['low', 'high', 'volume']
    for lag_i in lag:
        for parametro_i in parametro:
            nombreFeature = "lag" + parametro_i + str(lag_i)
            df[nombreFeature] = (computelag(dataframe[parametro_i], lag_i) - dataframe[parametro_i]) / dataframe[
                parametro_i]
    return df


def anadirFearAndGreed(dataframe):
    df = dataframe
    df['feargreed'] = computeFearAndGreed(dataframe)
    return df


def anadirEMARelativa(dataframe):
    df = dataframe
    periodo = [5, 10, 15, 50]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodo:
        for parametro_i in parametro:
            nombreFeature = "ema" + parametro_i + str(periodo_i)
            df[nombreFeature] = (calculate_ema(dataframe[parametro_i], periodo_i) - dataframe[parametro_i]) / dataframe[
                parametro_i]
    return df


def anadirSMARelativa(dataframe):
    df = dataframe
    periodos = [5, 10, 15, 25, 50]
    parametro = ['adjclose', 'volume']
    for periodo_i in periodos:
        for parametro_i in parametro:
            nombreFeature = "sma" + parametro_i + str(periodo_i)
            df[nombreFeature] = (calculate_sma(dataframe[parametro_i], periodo_i) - dataframe[parametro_i]) / dataframe[
                parametro_i]
    return df


def anadirHammerRangosRelativa(dataframe):
    df = dataframe
    diasPreviosA = [1, 2, 3, 10, 15, 20]
    diasPreviosB = [1, 2, 3, 10, 15, 20]
    parametroA = ['high', 'low', 'volume']
    parametroB = ['high', 'low', 'volume']
    parametroC = ['high', 'low', 'volume']

    for diasPreviosA_i in diasPreviosA:
        for diasPreviosB_i in diasPreviosB:
            for parametroA_i in parametroA:
                for parametroB_i in parametroB:
                    for parametroC_i in parametroC:
                        nombreFeature = "hammer" + str(diasPreviosA_i) + "y" + str(
                            diasPreviosB_i) + parametroA_i + parametroB_i + parametroC_i
                        df[nombreFeature] = calculadoraHammerRelativa(data=dataframe, diasPreviosA=diasPreviosA_i,
                                                                      diasPreviosB=diasPreviosB_i,
                                                                      parametroA=parametroA_i,
                                                                      parametroB=parametroB_i,
                                                                      parametroC=parametroC_i)

    return df


def anadirvwapRelativa(dataframe):
    df = dataframe
    parametroA = ['adjclose']
    parametroB = ['volume']
    for parametroA_i in parametroA:
        for parametroB_i in parametroB:
            nombreFeature = "vwap" + parametroA_i + parametroB_i
            df[nombreFeature] = computevwapRelativa(df, parametroA_i, parametroB_i)
    return df


def anadirDistanciaAbollingerRelativa(dataframe):
    df = dataframe
    parametroA = [5, 10, 30]  # datapoint rolling window. DEBE SER MAYOR QUE 1 SIEMPRE
    parametroB = [5, 10, 30]  # sigma width. DEBE SER MAYOR QUE 1 SIEMPRE
    parametroC = ['adjclose', 'volume']
    for parametroA_i in parametroA:
        for parametroB_i in parametroB:
            for parametroC_i in parametroC:
                nombreFeatureMA = "bollingerMA" + str(parametroA_i) + "-" + str(parametroB_i) + parametroC_i
                nombreFeatureBU = "bollingerBU" + str(parametroA_i) + "-" + str(parametroB_i) + parametroC_i
                nombreFeatureBL = "bollingerBL" + str(parametroA_i) + "-" + str(parametroB_i) + parametroC_i
                MA, BU, BL = computebollinger_bands(dataframe, parametroA_i, parametroB_i)
                df[nombreFeatureMA] = 1 + 100000 * (distanciaTantoPorUno(df[parametroC_i], MA) - df[parametroC_i]) / df[
                    parametroC_i]
                df[nombreFeatureBU] = 1 + 100000 * (distanciaTantoPorUno(df[parametroC_i], BU) - df[parametroC_i]) / df[
                    parametroC_i]
                df[nombreFeatureBL] = 1 + 100000 * (distanciaTantoPorUno(df[parametroC_i], BL) - df[parametroC_i]) / df[
                    parametroC_i]
    return df


def anadirATR(dataframe):
    df = dataframe
    periodos = [5, 10, 15, 20]
    for periodo_i in periodos:
        nombreFeature = "atr" + str(periodo_i)
        df[nombreFeature] = computeATR(dataframe, periodo_i)
    return df


def anadirCCI(dataframe):
    df = dataframe
    periodos = [5, 10, 15, 25, 40, 50]
    for periodo_i in periodos:
        nombreFeature = "cci" + str(periodo_i)
        df[nombreFeature] = computeCCI(dataframe, periodo_i)
    return df


#  Variación de la high positiva en x días (periodo acumulado de salto) respecto de la mediana,
#  amplificado por el volumen relativo
def anadirsupernovaTipoA(dataframe):
    df = dataframe

    # Periodos
    periodo = [1, 2, 3, 5, 10, 20]

    # salto high (variación en 1 día)
    highDesplazado1 = computeDerivadaDesfase(df['high'], 1)
    variacionHigh1 = df['high'] - highDesplazado1  # salto de 1 día
    variacionHigh1mediana = computeMedian(variacionHigh1)  # mediana del salto

    # Variación relativa de volumen
    mediaVolumen = computeMedian(df['volume'])
    volumenRelativo = df['volume'] / mediaVolumen

    for periodo_i in periodo:
        # salto high en x días (salto de x días)
        highDesplazadoX = computeDerivadaDesfase(df['high'], periodo_i)
        variacionHighX = df['high'] - highDesplazadoX  # salto de x días (salto de x días)
        variacionRelativaHighX = variacionHighX / variacionHigh1mediana  # Tamaño relativo de salto de x días respecto de la mediana de salto

        nombreFeature = "supernovaTipoA-" + str(periodo_i)
        df[nombreFeature] = variacionRelativaHighX * volumenRelativo

    return df


#  High positiva en A días respecto de la mediana del high en ese periodo,
#  amplificado por el volumen relativo en B días.
#  Si el high relativo es negativo, se reemplazará con 0.
def anadirsupernovaTipoB(dataframe):
    df = dataframe

    # Periodos
    periodoA = [1, 2, 3, 4, 5, 10, 20]
    periodoB = [1, 2, 3, 4, 5, 10, 20]

    for periodoA_i in periodoA:
        for periodoB_i in periodoB:
            # high relativa en A días
            highMedianaA = computeMedian(df['high'][:periodoA_i])  # mediana del high
            highRelativaA = df['high'] / highMedianaA
            highRelativaA = highRelativaA.clip(lower=0)

            # Volumen relativo en B días
            mediaVolumenB = computeMedian(df['volume'][:periodoB_i])
            volumenRelativoB = df['volume'] / mediaVolumenB

            nombreFeature = "supernovaTipoB-" + str(periodoA_i) + "-" + str(periodoB_i)
            df[nombreFeature] = highRelativaA * volumenRelativoB

    return df


#  Idea: high muy por encima de media en periodo grande (20 días: B) (tomar valor relativo).
#  Además, vela de subida fuerte en periodo pequeño (4 días: A) (es decir, high máximo de hace A días por
#  encima de media de A días), seguida de vela roja de freno fuerte (es decir, high muy por debajo del máximo)
#  y 2 velas verdes
#  (velas de 0 y 1 días antes) que no superan el máximo, ni bajan del mínimo.
#  Amplificado por volumen relativo fuerte en periodo pequeño frente periodo grande (B).
#   Si no de da la orientación esperada, multiplicar por 0 en cada parámetro
def anadirsupernovaTipoC(dataframe):
    df = dataframe

    # Periodos
    periodoA = [1, 2, 3, 4, 5, 8, 15]
    periodoB = [1, 2, 3, 4, 5, 8, 15]

    for periodoA_i in periodoA:
        for periodoB_i in periodoB:
            # high relativa en A días
            highMedianaA = computeMedian(df['high'][:periodoA_i])  # mediana del high
            highRelativaA = df['high'] / highMedianaA
            highRelativaA = highRelativaA.clip(lower=0)

            # high máxima en A días
            highMaxA = computeMaximo(df['high'][:periodoA_i])  # max del high
            highMaxRelativaA = df['high'] / highMedianaA
            highMaxRelativaA = highMaxRelativaA.clip(lower=0)

            # high relativa en B días
            highMedianaB = computeMedian(df['high'][:periodoB_i])  # mediana del high
            highRelativaB = df['high'] / highMedianaB
            highRelativaB = highRelativaB.clip(lower=0)

            # Vela de hace 4 días
            adjcloseDesplazado4 = computeDerivadaDesfase(df['adjclose'], 4)
            openDesplazado4 = computeDerivadaDesfase(df['open'], 4)
            vela4 = adjcloseDesplazado4 - openDesplazado4  # Vela de hace 4 días
            vela4 = vela4.clip(lower=0)

            # Vela de hace 1 días
            adjcloseDesplazado1 = computeDerivadaDesfase(df['adjclose'], 1)
            openDesplazado1 = computeDerivadaDesfase(df['open'], 1)
            vela1 = adjcloseDesplazado1 - openDesplazado1  # Vela de hace 1 días
            vela1 = vela1.clip(lower=0)

            # Vela de hace 0 días
            adjcloseDesplazado0 = computeDerivadaDesfase(df['adjclose'], 0)
            openDesplazado0 = computeDerivadaDesfase(df['open'], 0)
            vela0 = adjcloseDesplazado0 - openDesplazado0  # Vela de hace 0 días
            vela0 = vela0.clip(lower=0)

            # Volumen AB
            mediaVolumenA = computeMedian(df['volume'][:periodoA_i])
            mediaVolumenB = computeMedian(df['volume'][:periodoB_i])
            volumenRelativoAB = mediaVolumenA / mediaVolumenB
            if volumenRelativoAB < 0:
                volumenRelativoAB = 0

            nombreFeature = "supernovaTipoC-" + str(periodoA_i) + "-" + str(periodoB_i)
            df[nombreFeature] = highRelativaB * highMaxRelativaA * vela4 * vela1 * vela0 * volumenRelativoAB

    return df


#  High positiva en x días respecto de la mediana del high,
#  amplificado por el volumen relativo en x días
def anadirsupernovaTipoD(dataframe):
    df = dataframe

    # Periodos
    periodo = [2, 5, 15, 30]

    for periodo_i in periodo:
        # high relativa en x días
        highMedianaX = computeMedian(df['high'][:periodo_i])  # mediana del high
        highRelativaX = df['high'] / highMedianaX

        # Volumen relativo en x días
        mediaVolumenX = computeMedian(df['volume'][:periodo_i])
        volumenRelativoX = df['volume'] / mediaVolumenX

        nombreFeature = "supernovaTipoD-" + str(periodo_i)
        df[nombreFeature] = highRelativaX * volumenRelativoX

    return df


# Se considerará indicador si hay velas de los últimos días con tamaño positivo creciente y grande relativo respecto
# del histórico. En volumen lo mismo.
# Si alguna de estas velas es negativa, se fijará a 0, para que el indicador completo sea 0.
# la vela de antigüedad 0 pesará el cuadrado frente a la de antigüedad 1.
def anadirsupernovaTipoE(dataframe):
    df = dataframe

    # Volumen medio
    mediaVolumen = computeMedian(df['volume'])

    # Velas
    df['velaE'] = (df['adjclose'] - df['open'])

    # Velas relativas (valor absoluto, por velas negativas también)
    mediaVela = computeMedian(df['velaE'].abs())
    df['velarelativaE'] = df['velaE'] / mediaVela

    df['fuerzarelativaE-lag0'] = df['velarelativaE'] * df['volume'] / mediaVolumen

    # Velas días x
    df['fuerzarelativaE-lag0-clip'] = df['fuerzarelativaE-lag0'].clip(lower=0)
    df['fuerzarelativaE-lag1-clip'] = computeDerivadaDesfase(df['fuerzarelativaE-lag0'], 1).clip(lower=0)
    df['fuerzarelativaE-lag2-clip'] = computeDerivadaDesfase(df['fuerzarelativaE-lag0'], 2).clip(lower=0)
    df['fuerzarelativaE-lag5-clip'] = computeDerivadaDesfase(df['fuerzarelativaE-lag0'], 5).clip(lower=0)
    df['fuerzarelativaE-lag10-clip'] = computeDerivadaDesfase(df['fuerzarelativaE-lag0'], 10).clip(lower=0)

    df['fuerzarelativaElower0-1'] = df['fuerzarelativaE-lag0-clip'] * df['fuerzarelativaE-lag0-clip'] * df[
        'fuerzarelativaE-lag1-clip']
    df['fuerzarelativaElower0-2'] = df['fuerzarelativaE-lag0-clip'] * df['fuerzarelativaE-lag0-clip'] * df[
        'fuerzarelativaE-lag2-clip']
    df['fuerzarelativaElower0-5'] = df['fuerzarelativaE-lag0-clip'] * df['fuerzarelativaE-lag0-clip'] * df[
        'fuerzarelativaE-lag5-clip']
    df['fuerzarelativaElower0-10'] = df['fuerzarelativaE-lag0-clip'] * df['fuerzarelativaE-lag0-clip'] * df[
        'fuerzarelativaE-lag10-clip']

    return df


# Se considerará indicador si hay velas de los últimos días con tamaño negativo creciente y grande relativo respecto
# del histórico. En volumen lo mismo.
# Si alguna de estas velas es positiva, se fijará a 0, para que el indicador completo sea 0.
# la vela de antigüedad 0 pesará el cuadrado frente a la de antigüedad 1.
def anadirsupernovaTipoF(dataframe):
    df = dataframe

    # Volumen medio
    mediaVolumen = computeMedian(df['volume'])

    # Velas
    df['velaF'] = (df['adjclose'] - df['open'])

    # Velas relativas (valor absoluto, por velas negativas también)
    mediaVela = computeMedian(df['velaF'].abs())
    df['velarelativaF'] = df['velaF'] / mediaVela

    df['fuerzarelativaF-lag0'] = df['velarelativaF'] * df['volume'] / mediaVolumen

    # Vela día x
    df['fuerzarelativaF-lag0-clip'] = -(-df['fuerzarelativaF-lag0']).clip(lower=0)
    df['fuerzarelativaF-lag1-clip'] = -(-computeDerivadaDesfase(df['fuerzarelativaF-lag0'], 1)).clip(lower=0)
    df['fuerzarelativaF-lag2-clip'] = -(-computeDerivadaDesfase(df['fuerzarelativaF-lag0'], 2)).clip(lower=0)
    df['fuerzarelativaF-lag5-clip'] = -(-computeDerivadaDesfase(df['fuerzarelativaF-lag0'], 5)).clip(lower=0)
    df['fuerzarelativaF-lag10-clip'] = -(-computeDerivadaDesfase(df['fuerzarelativaF-lag0'], 10)).clip(lower=0)

    df['fuerzarelativaFlower0-1'] = df['fuerzarelativaF-lag0-clip'] * df['fuerzarelativaF-lag0-clip'] * df[
        'fuerzarelativaF-lag1-clip']
    df['fuerzarelativaFlower0-2'] = df['fuerzarelativaF-lag0-clip'] * df['fuerzarelativaF-lag0-clip'] * df[
        'fuerzarelativaF-lag2-clip']
    df['fuerzarelativaFlower0-5'] = df['fuerzarelativaF-lag0-clip'] * df['fuerzarelativaF-lag0-clip'] * df[
        'fuerzarelativaF-lag5-clip']
    df['fuerzarelativaFlower0-10'] = df['fuerzarelativaF-lag0-clip'] * df['fuerzarelativaF-lag0-clip'] * df[
        'fuerzarelativaF-lag10-clip']

    return df


def anadiraaron(dataframe):
    df = dataframe
    periodo = [5, 10, 15, 25, 50]
    for periodo_i in periodo:
        nombreFeatureUp = "aaron-up-" + str(periodo_i)
        nombreFeatureDown = "aaron-down-" + str(periodo_i)
        salida = computearoon(df, periodo_i)
        df[nombreFeatureUp] = salida[0]
        df[nombreFeatureDown] = salida[1]

    return df


def anadiradl(dataframe):
    df = dataframe
    nombreFeature = "adl"
    df[nombreFeature] = computeadl(data=df, high_col="high", low_col="low", close_col="adjclose", volume_col="volume")
    return df


def anadirstochastic_oscillator(dataframe):
    df = dataframe
    periodo = [5, 10, 15]
    for periodo_i in periodo:
        nombreFeaturek = "stochastic-k-" + str(periodo_i)
        nombreFeatured = "stochastic-d-" + str(periodo_i)
        nombreFeaturekd = "stochastic-kd-" + str(periodo_i)
        salida = computestochastic_oscillator(df, N=periodo_i, M=3)
        df[nombreFeaturek] = salida[0]
        df[nombreFeatured] = salida[1]
        df[nombreFeaturekd] = salida[0] - salida[1]
    return df


def anadirVolumenRelativo(dataframe):
    df = dataframe
    # Variación relativa de volumen
    mediaVolumen = computeMedian(df['volume'])
    nombreFeature = "volumenrelativo"
    df[nombreFeature] = df['volume'] / mediaVolumen
    return df


def anadirFeaturesJapanCompetition1(dataframe):
    # https://www.kaggle.com/code/uioiuioi/2nd-place-solution
    df = dataframe

    periodo = [10, 20, 40, 60]  # No se puede poner un valor inferior a 3
    parametro = ['open', 'high', 'low', 'adjclose', 'volume']
    for periodo_i in periodo:
        for parametro_i in parametro:
            nombreFeatureRe = "return-" + str(periodo_i) + "-" + parametro_i
            nombreFeatureVol = "volatility-" + str(periodo_i) + "-" + parametro_i
            nombreFeatureMaGap = "MA-gap-" + str(periodo_i) + "-" + parametro_i
            df[nombreFeatureRe] = df[parametro_i].pct_change(periodo_i)
            df[nombreFeatureVol] = (np.log(df[parametro_i])).diff().rolling(periodo_i).std()
            df[nombreFeatureMaGap] = df[parametro_i] / (df[parametro_i].rolling(60).mean())

    return df


def anadirGapAcumulado(dataframe):
    df = dataframe
    periodoA = [0, 1, 2, 3, 4]
    parametroA1 = ['open', 'high', 'low', 'adjclose', 'volume']
    parametroA2 = ['open', 'high', 'low', 'adjclose', 'volume']
    for periodoA_i in periodoA:
        for parametroA1_i in parametroA1:
            for parametroA2_i in parametroA2:
                nombreFeature = "gapacum-" + str(periodoA_i) + "-" + parametroA1_i + "-" + parametroA2_i
                df[nombreFeature] = calculadoraGap(df, diasPreviosA=periodoA_i,
                                                   parametroA1=parametroA1_i,
                                                   parametroA2=parametroA2_i)
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


# Calculadora de desviación típica leyendo el parámetro INCREMENTO
def computeDesviacionTipicaFromIncremento(data):
    return data['INCREMENTO'].std()


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
def calculadoraHammerRelativa(data, diasPreviosA, diasPreviosB, parametroA="open", parametroB="low",
                              parametroC="adjclose"):
    # Se calculará en tanto por uno la fuerza del patrón martillo, según:
    # Hammer = caída inicial (valor positivo si cae) * subida final (valor positivo si sube).
    # Si no es caída, sino lo inverso, se fijará a 0.
    # La caída inicial será: (parametroB  - parametroA), ambos los días previos indicados como parámetro
    # La subida final será: (parametroC - parametroB), donde low será el día previo indicado como parámetro, y el adjclose será de hoy
    # De todo se toma el valor relativo al mediano
    a = data[parametroA] / computeMedian(data[parametroA])
    b = data[parametroB] / computeMedian(data[parametroB])
    c = data[parametroC] / computeMedian(data[parametroC])
    aDesplazado = a.shift(diasPreviosA)
    bDesplazado = b.shift(diasPreviosB)

    caidaInicial = (bDesplazado - aDesplazado) / aDesplazado
    caidaInicial = caidaInicial.clip(lower=0)
    subidaFinal = (c - bDesplazado) / c
    # subidaFinal = subidaFinal.clip(lower=0)
    hammer = caidaInicial * subidaFinal
    return hammer


def computevwapRelativa(data, parametroA="adjclose", parametroB="volume"):
    df = data
    # https://altcoinoracle.com/calculate-the-volume-weighted-average-price-vwap-in-python/
    # Calculate the cumulative total of price times volume
    ab = df[parametroA] * df[parametroB]
    cumab = ab.cumsum()

    # Calculate the cumulative total of volume
    cumb = df[parametroB].cumsum()

    # Calculate VWAP relativa
    return (cumab - cumb) / cumb


def computebollinger_bands(dataframe, n, m):
    df = dataframe
    # https://tcoil.info/compute-bollinger-bands-for-stocks-with-python-and-pandas/
    # takes dataframe on input
    # n = smoothing length
    # m = number of standard deviations away from MA

    # typical price
    TP = (df['high'] + df['low'] + df['adjclose']) / 3
    # but we will use Adj close instead for now, depends

    data = TP
    # data = df['Adj Close']

    # takes one column from dataframe
    B_MA = pd.Series((data.rolling(n, min_periods=n).mean()), name='B_MA')
    sigma = data.rolling(n, min_periods=n).std()

    BU = pd.Series((B_MA + m * sigma), name='BU')
    BL = pd.Series((B_MA - m * sigma), name='BL')

    return B_MA, BU, BL


def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1 / n, adjust=False).mean()


def computeATR(dataframe, n):
    df = dataframe
    data = df.copy()
    high = data['high']
    low = data['low']
    close = data['close']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    return atr


# Commodity Channel Index
def computeCCI(dataframe, n, constant=0.015):
    """
    Calculates the Commodity Channel Index (CCI) for a given set of prices and period

    Parameters:
    prices (list or np.array): A list or array of historical prices
    n (int): The number of periods to calculate the CCI over
    constant (float): The constant used to calculate the mean deviation

    Returns:
    list: A list of CCI values for each period
    """

    # Calculate typical price
    tp = (dataframe['high'] + dataframe['low'] + dataframe['adjclose']) / 3

    # Calculate the moving average of the typical price
    ma_tp = tp.rolling(n).mean()

    # Calculate the mean deviation
    md = np.abs(tp - ma_tp).rolling(n).mean()

    # Calculate the CCI
    cci = (tp - ma_tp) / (constant * md)

    return cci


# Aroon oscilator
def computearoon(data, period=14):
    aroon_up = []
    aroon_down = []

    for i in range(period, len(data)):
        highest_high = max(data[i - period:i]['high'])
        lowest_low = min(data[i - period:i]['low'])

        aroon_up.append((period - (i - data[i - period:i]['high'].index(highest_high))) / period * 100)
        aroon_down.append((period - (i - data[i - period:i]['low'].index(lowest_low))) / period * 100)

    return aroon_up, aroon_down


def computestochastic_oscillator(data, N=14, M=3):
    df = data
    df['low-' + str(N)] = df['low'].rolling(N).min()
    df['high-' + str(N)] = df['high'].rolling(N).max()
    df['K-' + str(N)] = 100 * (df['adjclose'] - df['low-' + str(N)]) / \
                        (df['high-' + str(N)] - df['low-' + str(N)])
    df['D-' + str(N)] = df['K-' + str(N)].rolling(M).mean()
    return df['K-' + str(N)], df['D-' + str(N)]


def computeMaximo(data):
    return data.max()


def computeMinimo(data):
    return data.min()


def computeMedian(data):
    return data.median()


def distanciaTantoPorUno(A, B):
    return (B - A) / A


def tomarprimerosdatos(data, tamano):
    return data[:tamano]


def computeDerivadaDesfase(data, desfase):
    return data.diff(desfase)


# Obtiene el ADL (Accumulation Distribution Line indicator)
def computeadl(data: pd.DataFrame, high_col: str, low_col: str, close_col: str, volume_col: str) -> pd.Series:
    """
    Calculates the Accumulation/Distribution Line (ADL) for a given DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the historical data
    high_col : str
        Name of the column containing the high price
    low_col : str
        Name of the column containing the low price
    close_col : str
        Name of the column containing the close price
    volume_col : str
        Name of the column containing the volume

    Returns
    -------
    pd.Series
        The Accumulation/Distribution Line for the given DataFrame
    """
    # Calculate money flow multiplier
    data['mfm'] = ((data[close_col] - data[low_col]) - (data[high_col] - data[close_col])) / (
            data[high_col] - data[low_col])
    # Calculate money flow volume
    data['mfv'] = data['mfm'] * data[volume_col]
    # Calculate the Accumulation/Distribution Line
    adl = data['mfv'].cumsum()
    return adl


def calculadoraGap(data, diasPreviosA=1, parametroA1="open", parametroA2="adjclose"):
    # Se toman los parámetros
    a1 = data[parametroA1]
    a2 = data[parametroA2]

    a1Desplazado = a1.shift(diasPreviosA)
    a2Desplazado = a2.shift(diasPreviosA + 1)

    gap = (a1Desplazado - a2Desplazado) / a2Desplazado

    gapAcumulado = gap
    return gapAcumulado


def generaModeloLightGBM(datos, metrica, pintarFeatures=False, pathCompletoDibujoFeatures="", carpeta=""):
    # Aleatorización de datos
    df = aleatorizarDatos(datos)

    # Se eliminan datos no numéricos
    df = clean_dataset(df)

    # Se trocean los datos para entrenar y validar
    X_train, X_test, y_train, y_test = divideDatosParaTrainTestXY(df)

    # Imbalanced data: se transforma el dataset
    # transform the dataset
    smote = SMOTEENN()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    params = {'objective': 'binary',
              'learning_rate': 0.01,
              "boosting_type": "gbdt",
              "metric": metrica,
              'n_jobs': -1,
              'min_data_in_leaf': 32,
              'num_leaves': 1024,
              }
    model = lgb.LGBMClassifier(**params, n_estimators=100)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)])

    # DEBUG:
    y_pred_train = model.predict(X_train)
    print("SE DEBE OPTIMIZAR LA PRECISIÓN Y EL RECALL. Esto es con el train, asi que debería salir 100%:")
    print('Training-set LGB precision score: {0:0.4f}'.format(precision_score(y_train, y_pred_train)))
    print('Training-set LGB recall score: {0:0.4f}'.format(recall_score(y_train, y_pred_train)))

    # DEBUG:
    y_pred_test = model.predict(X_test)
    print("Para los datos de VALIDACIÓN, básico, sin proba:")
    print('Precision score: {0:0.4f}'.format(precision_score(y_test, y_pred_test)))
    print('Recall score: {0:0.4f}'.format(recall_score(y_test, y_pred_test)))

    # DEBUG:
    if pintarFeatures:
        #################### SE DIBUJAN LAS FEATURES POR IMPORTANCIA #################
        columns = [col for col in X_train.columns]
        feature_importance = pd.DataFrame()
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = columns
        fold_importance["importance"] = model.feature_importances_
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        # Se pintan las primeras x features
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
               :10].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
        features_ordenadas = best_features.sort_values(by="importance", ascending=False)
        print("Features por importancia:")
        print(features_ordenadas)
        # export DataFrame to text file (keep header row and index column)
        colsCompletas = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance",
            ascending=False).index

        best_featuresCompletas = feature_importance.loc[feature_importance.feature.isin(colsCompletas)]
        features_ordenadasCompletas = best_featuresCompletas.sort_values(by="importance", ascending=False)
        with open(carpeta + "features_ordenadasCompletas.txt", 'w') as f:
            df_string = features_ordenadasCompletas.to_string()
            f.write(df_string)
            f.close()

        plt.figure(figsize=(16, 12));
        sns.barplot(x="importance", y="feature", data=features_ordenadas);
        plt.title('Importance of Feature');
        # plt.show(block=False)
        # plt.pause(5)
        plt.savefig(pathCompletoDibujoFeatures)
        # plt.close()
        ###################

    # DEBUG:
    # visualize confusion matrix with seaborn heatmap
    cm = confusion_matrix(y_test, y_pred_test)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])
    svm = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    figure = svm.get_figure()
    figure.savefig(carpeta + 'heatmapAlGenerarModeloLightGBM.png', dpi=400)

    # DEBUG:
    calculaPrecision(y_test, y_pred_test)

    # DEBUG:
    umbral = 0.6
    pred, proba = predictorConProba(model, X_test, umbralProba=umbral, analizarResultado=True,
                                    y_solucionParaAnalisis=y_test,
                                    mensajeDebug="Análisis en la creación el modelo mirando la probabilidad de TARGET==1 y filtrando por proba: " + str(
                                        umbral))

    return model


def calculaPrecision(target, prediccion):
    # PRECISION (TP/(TP+FP))
    TP = sum(1 for x, y in zip(target, prediccion) if (x == y and y == 1))
    TPandFP = sum(prediccion)
    precision = TP / TPandFP

    # DEBUG
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    print("TP: ", TP)
    print("TP + FP: ", TPandFP)
    print("PRECISION: ", precision)
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(target, prediccion, digits=3))

    return precision


def evaluate(model, testing_set_x, testing_set_y):
    predictions = model.predict_proba(testing_set_x)

    accuracy = accuracy_score(testing_set_y, predictions[:, 1] >= 0.5)
    roc_auc = roc_auc_score(testing_set_y, predictions[:, 1])
    precision = precision_score(testing_set_y, predictions[:, 1] >= 0.5)
    recall = recall_score(testing_set_y, predictions[:, 1] >= 0.5)
    pr_auc = average_precision_score(testing_set_y, predictions[:, 1])

    result = pd.DataFrame([[accuracy, precision, recall, roc_auc, pr_auc]],
                          columns=['Accuracy', 'Precision', 'Recall', 'ROC_auc', 'PR_auc'])
    return (result)


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)

    # Se quitan las siguientes columnas, sólo si existen:
    if set(['date', 'ticker']).issubset(df.columns):
        df = df.drop(columns=['date', 'ticker'])

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)


def divideDatosParaTrainTestXY(datos):
    # Se separan features de TARGET, y se quitan columnas no numéricas
    X, y = limpiaDatosParaUsarModelo(datos)
    print("tamano: ", str(X.size) + " x " + str(y.size))
    X_A, X_B, y_A, y_B = train_test_split(X, y, stratify=y, test_size=0.5)
    return X_A, X_B, y_A, y_B


def limpiaDatosParaUsarModelo(datos):
    # Se separan features de TARGET, y se quitan columnas no numéricas
    # Se devuelven las columnas de features limpias y target
    X = datos.drop(columns=['TARGET'])
    X = X.drop(columns=['INCREMENTO'])
    X = X.drop(columns=['diff'])
    y = datos['TARGET']
    return X, y


def troceaDataframeMismoTamano(datos, numPartesIguales):
    # Se reparten las filas entre los dataframes indicados. No se hace aleatorización previa.
    # Se revuelven tantos dataframes del mismo tamaño como se indique como parámetro
    return np.array_split(datos, numPartesIguales)


def predictorConProba(modelo, X, umbralProba=0.8, analizarResultado=False, y_solucionParaAnalisis=pd.DataFrame(),
                      mensajeDebug=""):
    # Se intenta subir la precisión en base a coger sólo las predicciones de alta probabilidad
    # Parámetros:
    # Modelo: modelo ya entrenado
    # X: features limpias
    # y_validacion: Opcionalmente
    # Se devuelve la predicción y su probabilidad

    if analizarResultado == True:
        aux = pd.concat([X, y_solucionParaAnalisis], axis=1)
        aux = clean_dataset(aux)
        auxSinTarget = aux.drop(columns=['TARGET'])
    else:
        aux = X
        aux = clean_dataset(aux)
        auxSinTarget = aux

    # Se analizan sólo las predicciones a 1
    aux["proba"] = modelo.predict_proba(auxSinTarget)[:, 1]
    threshold = umbralProba
    aux["pred"] = aux["proba"].apply(lambda el: 1.0 if el >= threshold else 0.0)

    print("Umbral de probabilidad (umbralProba): ", umbralProba)

    if analizarResultado == True:
        print(mensajeDebug)
        print("MODIFICAR EL UMBRAL de 0.5 LLEVA A OVERFITTING")
        auxSoloTarget = aux['TARGET']
        print('Precision score: {0:0.4f}'.format(precision_score(auxSoloTarget, aux["pred"])))
        print('Recall score: {0:0.4f}'.format(recall_score(auxSoloTarget, aux["pred"])))
        calculaPrecision(auxSoloTarget, aux["pred"])

    return aux["pred"], aux["proba"]


def creaModelo(filepathModeloAGuardar, descargarInternetParaGenerarModelo=True):
    print("----------------------------------------------------------")
    print("--- GENERADOR DE MODELO -----")
    print("----------------------------------------------------------")

    # Descarga de la información
    if descargarInternetParaGenerarModelo:
        descargaDatosACsv(cuantasEmpresas, startDate, endDate, carpeta, nombreFicheroCsvBasica,
                          indiceComienzoListaEmpresas)

    # Crear parámetros avanzados y target
    procesaInformacion(carpeta, nombreFicheroCsvBasica, carpeta, nombreFicheroCsvAvanzado)

    # Leer fichero
    datos = leerCSV(carpeta + nombreFicheroCsvAvanzado)

    # Se trocean los datos para train/test y validación
    datosTrainTest, datosValidacion = troceaDataframeMismoTamano(datos, 2)

    # Generación de modelo ya evaluado
    modelo = generaModeloLightGBM(datos=datosTrainTest, metrica="binary_logloss", pintarFeatures=True,
                                  pathCompletoDibujoFeatures=carpeta + featuresporimportancia, carpeta=carpeta)

    # Se valida el modelo con datos independientes
    X_valid, y_valid = limpiaDatosParaUsarModelo(datosValidacion)
    y_pred_valid, y_proba_valid = predictorConProba(modelo, X_valid, umbralProba=0.5, analizarResultado=True,
                                                    y_solucionParaAnalisis=y_valid,
                                                    mensajeDebug="Análisis con datos INDEPENDIENTES (VALIDACIÓN) y usando la proba para predecir: ")

    # Se añaden las columnas de predicción y probabilidad
    datosValidacion = datosValidacion.join(y_pred_valid)
    datosValidacion = datosValidacion.join(y_proba_valid)

    # Análisis de sólo las filas donde invertir
    y_pred_a_invertir_valid = y_pred_valid[y_pred_valid == 1]
    datos_a_invertir = datosValidacion[datosValidacion.index.isin(y_pred_a_invertir_valid.index)]
    rentaMedia_valid = computeRentabilidadMediaFromIncremento(datos_a_invertir)
    print("RENTA MEDIANA DE VALIDACIÓN: ", rentaMedia_valid)

    joblib.dump(modelo, filepathModeloAGuardar)

    return modelo


def predecir(pathModelo, umbralProba=0.5, necesitaDescarga=True):
    # Se carga el modelo de predicción, ya entrenado y validado
    modelo = joblib.load(pathModelo)

    # Descarga de la información
    if necesitaDescarga:
        descargaDatosACsv(PREDICCIONcuantasEmpresas, PREDICCIONstartDate, PREDICCIONendDate, carpeta,
                          PREDICCIONnombreFicheroCsvBasica, offsetEmpresas=PREDICCIONindiceComienzoListaEmpresas)

    # Crear parámetros avanzados y target
    procesaInformacion(carpeta, PREDICCIONnombreFicheroCsvBasica, carpeta, PREDICCIONnombreFicheroCsvAvanzado)

    # Leer fichero
    datos = leerCSV(carpeta + PREDICCIONnombreFicheroCsvAvanzado)

    # Se valida el modelo con datos independientes
    X_valid, y_valid = limpiaDatosParaUsarModelo(datos)
    y_pred_valid = []
    y_proba_valid = []
    if not X_valid.empty:
        y_pred_valid, y_proba_valid = predictorConProba(modelo, X_valid, umbralProba=umbralProba,
                                                        analizarResultado=False)

    # Análisis de sólo las filas donde invertir
    y_pred_a_invertir_valid = y_pred_valid[y_pred_valid == 1]
    datosValidacion_a_invertir_valid = datos[datos.index.isin(y_pred_a_invertir_valid.index)]
    pathDondeInvertir = carpeta + "umbral-" + str(umbralProba) + "-" + PREDICCIONNombreFicheroCSVDondeInvertir
    # Se ordena por fecha:
    datosValidacion_a_invertir_valid = ordenarPorFechaMasReciente(datosValidacion_a_invertir_valid)
    print("Fichero CSV con la información donde invertir: ", pathDondeInvertir)
    guardarDataframeEnCsv(datosValidacion_a_invertir_valid, pathDondeInvertir)

    # Análisis de rentas cercanas
    y_pred_a_invertir_valid = y_pred_valid[y_pred_valid == 1]
    datosValidacion_a_invertir_valid = datosValidacion_a_invertir_valid[
        datosValidacion_a_invertir_valid.index.isin(y_pred_a_invertir_valid.index)]

    # Se guardan las rentas en un fichero
    pathRentasDondeInvertir = carpeta + "rentasDondeInvertir-umbral-" + str(umbralProba) + ".txt"
    with open(pathRentasDondeInvertir, 'w') as f:
        incrementos = datosValidacion_a_invertir_valid['INCREMENTO']
        df_string = incrementos.to_string()
        f.write(df_string)
        f.close()

    rentaMedia_valid = computeRentabilidadMediaFromIncremento(datosValidacion_a_invertir_valid)
    std_valid = computeDesviacionTipicaFromIncremento(datosValidacion_a_invertir_valid)
    print("...")

    print("umbral de probabilidad para escoger el target: ", umbralProba)
    print("RENTA MEDIANA de antigüedades cercanas (ligeramente antiguas) similares a donde invertir: ",
          rentaMedia_valid)
    print("DESVIACION TIPICA de antigüedades cercanas (ligeramente antiguas) similares a donde invertir: ",
          std_valid)

    return pathDondeInvertir


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


#################################################
#################################################
# SE IGNORAN LOS WARNINGS, PARA NO DESBORDAR LOS LOGS DE KAGGLE
#################################################
#################################################
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

    if ENTRENAR:
        #################################################
        #################################################
        # GENERADOR DE MODELO Y SU VALIDACIÓN
        #################################################
        #################################################
        # Se genera el modelo (con validación interna)):
        modelo = creaModelo(pathModelo, descargarInternetParaGenerarModelo=descargarInternetParaGenerarModelo)

    #################################################
    #################################################
    # PREDICTOR
    #################################################
    #################################################
    print("#################################################")
    print("#################################################")
    print("##########  PREDICTOR ##############")
    print("#################################################")
    print("#################################################")
    print("----------------------------------------------------------")
    print("--- COMIENZO DE PREDICCIÓN PARA INVERTIR DINERO REAL-----")
    print("----------------------------------------------------------")

    # Se imprime la fecha y hora actual en Madrid
    import pytz
    from datetime import datetime

    # initialize the local time
    l_time = datetime.now()
    # Conversion of loctime - GMT
    g_timezone = pytz.timezone('Europe/Madrid')
    g_time = l_time.astimezone(g_timezone)
    print("INSTANTE DE EJECUCIÓN en MADRID: ", g_time)

    # Se predice:
    umbral = 0.8
    print("Predicción con umbral: " + str(umbral))
    predecir(pathModelo, umbralProba=umbral, necesitaDescarga=True & descargarInternetParaGenerarModelo)
    print("#################################################")
    umbral = 0.7
    print("Predicción con umbral: " + str(umbral))
    predecir(pathModelo, umbralProba=umbral, necesitaDescarga=False & descargarInternetParaGenerarModelo)
    print("#################################################")
    umbral = 0.6
    print("Predicción con umbral: " + str(umbral))
    predecir(pathModelo, umbralProba=umbral, necesitaDescarga=False & descargarInternetParaGenerarModelo)
    print("#################################################")
    umbral = 0.4
    print("Predicción con umbral: " + str(umbral))
    predecir(pathModelo, umbralProba=umbral, necesitaDescarga=False & descargarInternetParaGenerarModelo)
    print("#################################################")
    umbral = 0.5
    print("Predicción con umbral: " + str(umbral))
    predecir(pathModelo, umbralProba=umbral, necesitaDescarga=False & descargarInternetParaGenerarModelo)
######################################################################

print("...")
print("FIN")