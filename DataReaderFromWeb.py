from random import randint
from time import sleep
import pandas as pd

from stock_info import get_data, tickers_sp500, tickers_nasdaq, tickers_other, get_quote_table


# empresasMaximas: NÚMERO MÁXIMO DE EMPRESAS A LEER. Si quiero todos, poner un número gigante: 999999
def getEmpresasFromNasdaq(empresasMaximas, startDate, endDate, offsetEmpresasNasdaq):
    empresasMaximasAux = empresasMaximas

    # Listado completo de empresas del NASDAQ
    nasdaq_ticker_list = tickers_nasdaq()

    # Se eliminan las N primeras, para poder escoger unas empresas u otras
    nasdaq_ticker_list = nasdaq_ticker_list[offsetEmpresasNasdaq:]

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

    # Se obtienen todos los datos del NASDAQ
    primero = True
    for i in range(len(nasdaq_ticker_list)):
        empresasMaximasAux = empresasMaximasAux - 1
        if empresasMaximasAux < 0:
            break

        # Esperar aleatoria x segundos
        sleep(randint(1, 4))

        # DEBUG
        print("Empresa "+str(i+1)+" - Se obtienen los datos de la empresa: ", nasdaq_ticker_list[i])

        try:
            datosEmpresa = get_data(nasdaq_ticker_list[i], start_date=startDate,
                                    end_date=endDate, index_as_date=False)
            if primero:
                primero = False
                # Dataframe vacío
                datoscompletos = pd.DataFrame(datosEmpresa)
            else:
                datoscompletos = datoscompletos.append(datosEmpresa, ignore_index=True)

        except AssertionError as error:
            print("No se ha podido obtener información de la empresa \"" + nasdaq_ticker_list[
                i] + "\" pero se continúa...")

    return datoscompletos
