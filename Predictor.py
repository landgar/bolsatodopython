from datetime import date
from datetime import timedelta
import joblib

from CreadorInformacion import *
from ProcesadorInformacion import *
from FabricadordeVariosModelos import *

pathModelo="/home/t151521/Descargas/prueba/MODELO.scikit"




#################################################
#################################################
# PREDICTOR
#################################################
#################################################
def predecir(pathModelo):
    carpeta = "/home/t151521/Descargas/prueba/"
    PREDICCIONnombreFicheroCsvBasica = "PREDICCIONinfosucio.csv"
    PREDICCIONnombreFicheroCsvAvanzado = "PREDICCIONinfolimpioavanzadoTarget.csv"
    PREDICCIONfeaturesporimportancia = "PREDICCIONfeaturesporimportancia.png"
    PREDICCIONNombreFicheroCSVDondeInvertir = "PREDICCIONdondeinvertir.csv"
    # Se toman 51 días hacia atrás, hasta ayer (para poder calcular RSI y demás)
    startDate = date.today() - timedelta(days=100)
    endDate = date.today() - timedelta(days=1)
    cuantasEmpresas = 100
    indiceComienzoListaEmpresasNasdaq = 3000

    print("----------------------------------------------------------")
    print("--- COMIENZO DE PREDICCIÓN PARA INVERTIR DINERO REAL-----")
    print("----------------------------------------------------------")

    # Se carga el modelo de predicción, ya entrenado y validado
    modelo = joblib.load(pathModelo)

    # Descarga de la información
    descargaDatosACsv(cuantasEmpresas, startDate, endDate, carpeta, PREDICCIONnombreFicheroCsvBasica,
                      offsetEmpresasNasdaq=indiceComienzoListaEmpresasNasdaq)

    # Crear parámetros avanzados y target
    procesaInformacion(carpeta, PREDICCIONnombreFicheroCsvBasica, carpeta, PREDICCIONnombreFicheroCsvAvanzado)

    # Leer fichero
    datos = leerCSV(carpeta + PREDICCIONnombreFicheroCsvAvanzado)

    # Se valida el modelo con datos independientes
    X_valid, y_valid = limpiaDatosParaUsarModelo(datos)
    y_pred_valid, y_proba_valid = predictorConProba(modelo, X_valid, umbralProba=0.5, analizarResultado=False)

    # Análisis de sólo las filas donde invertir
    y_pred_a_invertir_valid = y_pred_valid[y_pred_valid == 1]
    datosValidacion_a_invertir_valid = datos[datos.index.isin(y_pred_a_invertir_valid.index)]
    pathDondeInvertir = carpeta + PREDICCIONNombreFicheroCSVDondeInvertir
    # Se ordena por fecha:
    datosValidacion_a_invertir_valid = ordenarPorFechaMasReciente(datosValidacion_a_invertir_valid)
    print("Fichero CSV con la información donde invertir: ", pathDondeInvertir)
    guardarDataframeEnCsv(datosValidacion_a_invertir_valid, pathDondeInvertir)


    # Análisis de rentas cercanas
    y_pred_a_invertir_valid = y_pred_valid[y_pred_valid == 1]
    datosValidacion_a_invertir_valid = datosValidacion_a_invertir_valid[datosValidacion_a_invertir_valid.index.isin(y_pred_a_invertir_valid.index)]
    rentaMedia_valid = computeRentabilidadMediaFromIncremento(datosValidacion_a_invertir_valid)
    print("RENTA MEDIA DE antigüedades cercanas a donde invertir: ", rentaMedia_valid)


    return pathDondeInvertir


# Se predice:
predecir(pathModelo)

print("...FIN")

a = 1
