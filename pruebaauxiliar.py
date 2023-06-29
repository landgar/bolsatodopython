carpetaDestino = "/home/t151521/Descargas/prueba/"
fichero = "aux.csv"

from CreadorInformacion import *

def ordenarPorFechaMasReciente(data):
    return data.sort_values(by='date',ascending=False)

datos = leerCSV("/home/t151521/Descargas/prueba/"+"PREDICCIONdondeinvertir.csv")
datos=ordenarPorFechaMasReciente(datos)
guardarDataframeEnCsv(dataframe=datos, filepath=carpetaDestino+fichero)