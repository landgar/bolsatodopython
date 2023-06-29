# libraries
import pandas as pd
import os

# path = '/kaggle/input/stock-market-prediction/'
path = './'
pathOutput = './'
completoTrain = 'dataset_train_validation.csv'
completoTest = 'dataset_test.csv'

completoTrainLimpio = 'dataset_train_validationlimpio.csv'
completoTestLimpio = 'dataset_testlimpio.csv'

# INPUT
train = pd.read_csv(os.path.join(path, completoTrain))
test = pd.read_csv(os.path.join(path, completoTest))

# Los datasets de pruebas tienen algunas columnas con valores incorrectos (por la falta del punto decimal). se ajustará con el resto de valores de media


def limpiadatasets(df, columna):
    # Calcular la mediana de 'columna' para cada valor único en 'empresa'
    mediana_by_empresa = df.groupby('company')[columna].median()

    # Iterar sobre cada fila del DataFrame
    for index, row in df.iterrows():
        empresa = row['company']
        columnarow = row[columna]
        mediana = mediana_by_empresa[empresa]
        max_deviation = mediana * 1.0

        # Verificar si el valor de 'columna' se desvía más de un 100% de la mediana
        if columnarow > mediana + max_deviation or columnarow < mediana - max_deviation:
            while columnarow > mediana + max_deviation or columnarow < mediana - max_deviation:
                columnarow = columnarow / 10

        # Actualizar el valor de 'close' en la fila correspondiente
        df.at[index, columna] = columnarow

    return df


# Se procesa y guarda el resultado en un Excel
train=limpiadatasets(train, 'high')
train=limpiadatasets(train, 'low')
train=limpiadatasets(train, 'close')
train=limpiadatasets(train, 'open')
train.to_csv(os.path.join(path+ completoTrainLimpio), index=False)

test=limpiadatasets(test, 'high')
test=limpiadatasets(test, 'low')
test=limpiadatasets(test, 'close')
test=limpiadatasets(test, 'open')
test.to_csv(os.path.join(path+ completoTestLimpio), index=False)


