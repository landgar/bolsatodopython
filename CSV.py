import os
import pandas as pd

def guardarDataframeEnCsv(dataframe, filepath):
    os.makedirs('folder/subfolder', exist_ok=True)
    dataframe.to_csv(filepath, index=False)


def leerCSV(filepathCSV):
    return pd.read_csv(filepathCSV)