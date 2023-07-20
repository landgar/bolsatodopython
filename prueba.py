# PARA LEER LOS EARNINGS
# import nasdaqdatalink
#
# nasdaqdatalink.ApiConfig.api_key = "AHG1FNAFH5y9gf85ArpN"
# data = nasdaqdatalink.get_table('ZACKS/FC', paginate=True, ticker=['AAPL', 'MSFT'], per_end_date={'gte': '2015-01-01'})
#-------------------------------

#-------------------------------
# import quandl
# from datetime import datetime
# import pandas as pd
#
#
# quandl.ApiConfig.api_key = 'AHG1FNAFH5y9gf85ArpN'
# da=quandl.get_table('IFT/NSA', date='2023-06-01', ticker='AAPL')
# b=3
#
#
# quandl.ApiConfig.api_key = "AHG1FNAFH5y9gf85ArpN"
# startDate = '01/01/2022'
# endDate = '03/01/2022'
# nasdaq_ticker_list=['AAPL']
# tickersString = ','.join(map(str, nasdaq_ticker_list))
# startDate_d=datetime.strptime(startDate, "%d/%m/%Y")
# endDate_d=datetime.strptime(endDate, "%d/%m/%Y")
# date_list = pd.date_range(startDate_d,endDate_d, freq='D').strftime("%Y-%m-%d")
# listaDates = ','.join(map(str, date_list))
# data2=quandl.get_table('IFT/NSA', date='2018-05-01', ticker='AAPL')
# data = quandl.get_table('IFT/NSA', exchange_cd='US', date=listaDates, ticker=tickersString)
# a=1
# #-------------------------------
import pandas as pd
# pip install stockstats
from stockstats import StockDataFrame

data = pd.read_csv("/home/t151521/Descargas/prueba/infosucio.csv")
data.head()
data.rename(columns = {'Open':'open','High':'high','Low':'low',
                     'Close':'close','Volume':'volume'}, inplace=True)
stock = StockDataFrame.retype(data)

w=0


