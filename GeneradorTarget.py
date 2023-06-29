import pandas as pd

def anadirTarget(dataframe, minimoIncrementoEnPorcentaje, periodo):
    df=dataframe
    df['TARGET']=computeTargetConadjclose(df, minimoIncrementoEnPorcentaje, periodo)
    return df

def anadirIncrementoEnPorcentaje(dataframe, periodo):
    df=dataframe
    df['INCREMENTO']=computeIncrementoConadjclose(df, periodo)
    return df

def computeTargetConadjclose(data, incrementoEnPorcentaje, periodo):
    df=data
    # Se toma sólo la columna indicada para realizar los cálculos
    df['diff']=-data.groupby('ticker')['adjclose'].diff(-periodo)  # datos desplazados el periodo

    # Se calcula el target
    target = df.apply(lambda row: gettarget(row, incrementoEnPorcentaje), axis=1)
    return target

def computeIncrementoConadjclose(data, periodo):
    df=data
    # Se toma sólo la columna indicada para realizar los cálculos
    df['diff']=-data.groupby('ticker')['adjclose'].diff(-periodo)  # datos desplazados el periodo

    # Se calcula el incremento de subida
    incremento = df.apply(lambda row: getPercentage(row['diff'],row['adjclose']-row['diff']), axis=1)
    return incremento

def gettarget(row, incrementoEnPorcentaje):
    percentage = getPercentage(row['diff'],row['adjclose']-row['diff'])
    # Se asigna un 1 si se cumple el objetivo, y un 0 si no
    target = getBinary(percentage, incrementoEnPorcentaje)
    return target

def getPercentage(diferencia, base):
    try:
        percentage = (diferencia/base) * 100
    except ZeroDivisionError:
        percentage = float('-inf')
    return percentage


def getBinary(data, umbral):
    binary = data > umbral
    # Para convertir True/False en 1/0
    binary=binary*1
    return binary
