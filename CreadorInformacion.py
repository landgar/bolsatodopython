from DataReaderFromWeb import *
from CSV import *

# Guardar CSV con la información de las empresas
def descargaDatosACsv(cuantasEmpresas, startDate, endDate, carpeta, nombreFicheroCsv, offsetEmpresasNasdaq):
    datos=getEmpresasFromNasdaq(cuantasEmpresas, startDate, endDate, offsetEmpresasNasdaq)
    guardarDataframeEnCsv(dataframe=datos, filepath=carpeta+nombreFicheroCsv)

def ordenarPorFechaMasReciente(data):
    return data.sort_values(by='date',ascending=False)

