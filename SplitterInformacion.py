

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


