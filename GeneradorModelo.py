from datetime import date
from datetime import timedelta
import joblib

from CreadorInformacion import *
from ProcesadorInformacion import *
from FabricadordeVariosModelos import *

pathModelo="/home/t151521/Descargas/prueba/MODELO.scikit"

#################################################
#################################################
# GENERADOR DE MODELO Y SU VALIDACIÓN
#################################################
#################################################


def creaModelo(filepathModeloAGuardar):
    carpeta = "/home/t151521/Descargas/prueba/"
    nombreFicheroCsvBasica = "infosucio.csv"
    nombreFicheroCsvAvanzado = "infolimpioavanzadoTarget.csv"
    featuresporimportancia = "featuresporimportancia.png"
    startDate = '01/01/2022'
    endDate = '31/12/2022'
    cuantasEmpresas = 100
    indiceComienzoListaEmpresasNasdaq = 0

    print("----------------------------------------------------------")
    print("--- GENERADOR DE MODELO -----")
    print("----------------------------------------------------------")

    # Descarga de la información
    #descargaDatosACsv(cuantasEmpresas, startDate, endDate, carpeta, nombreFicheroCsvBasica, indiceComienzoListaEmpresasNasdaq)

    # Crear parámetros avanzados y target
    procesaInformacion(carpeta, nombreFicheroCsvBasica, carpeta, nombreFicheroCsvAvanzado)

    # Leer fichero
    datos = leerCSV(carpeta + nombreFicheroCsvAvanzado)

    # Se trocean los datos para train/test y validación
    datosTrainTest, datosValidacion = troceaDataframeMismoTamano(datos, 2)

    # Generación de modelo ya evaluado
    modelo = generaModeloLightGBM(datos=datosTrainTest, metrica="binary_logloss", pintarFeatures=True,
                                  pathCompletoDibujoFeatures=carpeta + featuresporimportancia)

    # Se valida el modelo con datos independientes
    X_valid, y_valid = limpiaDatosParaUsarModelo(datosValidacion)
    y_pred_valid, y_proba_valid = predictorConProba(modelo, X_valid, umbralProba=0.8, analizarResultado=True,
                                                    y_solucionParaAnalisis=y_valid,
                                                    mensajeDebug="Análisis con datos INDEPENDIENTES (VALIDACIÓN) y usando la proba para predecir: ")

    # Análisis de sólo las filas donde invertir
    y_pred_a_invertir_valid = y_pred_valid[y_pred_valid == 1]
    datosValidacion_a_invertir_valid = datosValidacion[datosValidacion.index.isin(y_pred_a_invertir_valid.index)]
    rentaMedia_valid = computeRentabilidadMediaFromIncremento(datosValidacion_a_invertir_valid)
    print("RENTA MEDIA DE VALIDACIÓN: ", rentaMedia_valid)

    joblib.dump(modelo, filepathModeloAGuardar)

    return modelo


# Se genera el modelo (con validación interna)):
modelo = creaModelo(pathModelo)
