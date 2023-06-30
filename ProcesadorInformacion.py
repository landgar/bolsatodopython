from CSV import *
from GeneradorParametrosAvanzados import *
from GeneradorTarget import *

def procesaInformacion(carpetaEntrada, nombreFicheroCsvEntrada, carpetaSalida, nombreFicheroCsvSalida):
    datos = leerCSV(carpetaEntrada + nombreFicheroCsvEntrada)
    # Se cuentan cuántos días de datos hay para cada empresa
    numFilasPorTicker=datos.pivot_table(columns=['ticker'], aggfunc='size')
    minimasFilasExigiblesPorTicker=max(numFilasPorTicker)

    # DEBUG:
    print("Filas por ticker: ", numFilasPorTicker)

    # DEBUG:
    print("Número mínimo de filas por ticket exigibles: ", minimasFilasExigiblesPorTicker)

    # Se eliminan los tickers con poca información (incompletos en días)
    numFilasPorTicker=numFilasPorTicker.to_frame()
    empresas = numFilasPorTicker[numFilasPorTicker[0] >= minimasFilasExigiblesPorTicker]
    datos=datos[datos['ticker'].isin(empresas.index)]

    # Se almacena el fichero limpio básico
    #nombreFicheroCsvSalidaBasico = "infolimpiobasico.csv"
    #guardarDataframeEnCsv(dataframe=datos, guardarIndice=False, filepath=carpetaSalida+nombreFicheroCsvSalidaBasico)

    datos=procesaEmpresas(datos)

    # # Si se añadiera la solución en las propias features, debería salir precisión=100%, y renta muy alta
    # datos['solucion']=datos['TARGET']

    # Se almacena el fichero avanzado con target
    guardarDataframeEnCsv(dataframe=datos, filepath=carpetaSalida+nombreFicheroCsvSalida)


def procesaEmpresa(datos):
    # Se añaden parámetros avanzados
    datos = anadirParametrosAvanzados(dataframe=datos)

    periodo=10

    # Se añade el incremento en porcentaje
    datos = anadirIncrementoEnPorcentaje(dataframe=datos, periodo=periodo)

    # Se añade el target
    datos = anadirTarget(dataframe=datos, minimoIncrementoEnPorcentaje=10, periodo=periodo)

    return datos

def procesaEmpresas(datos):
    primero = True
    empresas = datos.groupby("ticker")
    for nombreEmpresa, datosEmpresa in empresas:
        # DEBUG:
        print("Procesado de la empresa: ", nombreEmpresa)
        datos = procesaEmpresa(datosEmpresa)
        if primero:
            primero=False
            datoscompletos = pd.DataFrame(datosEmpresa)
        else:
            datoscompletos = datoscompletos.append(datosEmpresa, ignore_index=True)

    return datoscompletos


