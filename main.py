# libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import time
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import math

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_validate, StratifiedShuffleSplit
from sklearn import metrics

# DATA LOADING
from sklearn.neural_network import MLPClassifier

# path = '/kaggle/input/stock-market-prediction/'
path = './'
pathOutput = './'
completoTrain = 'dataset_train_validation.csv'
completoTest = 'dataset_test.csv'
#feardataFile = '/kaggle/input/crypto-fear-and-greed-index/fear_and_greed_index.csv'
feardataFile = 'fear_and_greed_index.csv'

# INPUT
train = pd.read_csv(os.path.join(path, completoTrain))
test = pd.read_csv(os.path.join(path, completoTest))
feardata = pd.read_csv(os.path.join(path, feardataFile))

# Se separan los resultados reales con los que comparar
columns = [col for col in test.columns if col not in ['company', 'age', 'market', 'TARGET']]
submission = test[columns]
solucion = test['TARGET']



# Se calcula la rentabilidad en 20 días, que es el periodo donde se calculó el target
# Desplazar los valores de "precio" 20 posiciones hacia atrás (PARA EL FUTURO)
test['close_20dias'] = test.groupby('company')['close'].shift(-20)
train['close_20dias'] = train.groupby('company')['close'].shift(-20)

# Calcular la rentabilidad en 20 días (objetivo del target)
def calcula_renta_porcentaje(row):
    if row['close_20dias'] is not None and row['close'] is not None:
        return (100*(row['close_20dias'] - row['close']))/row['close']
    else:
        return None

# Generación de la columna de renta
test['renta_20dias'] = test.apply(calcula_renta_porcentaje, axis=1)
train['renta_20dias'] = train.apply(calcula_renta_porcentaje, axis=1)

####################### ELIMINACION de filas con null en la columna TARGET
def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


train = filter_rows_by_values(train, "TARGET", ["null"])


########################## NUEVAS FEATURES ###########################
############ RSI ###################
def relative_strength_idx(df, n=14):
    close = df['close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


############ feargreedindex ###################
def feargreedindex(df, feardata):
    aux = df
    year = aux['year']
    month = aux['month']
    day = aux['day']
    aux['year-month-day'] = year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-" + day.astype(str).str.zfill(2)

    feardate = feardata['date']
    aux = aux.merge(feardata, left_on="year-month-day", right_on="date", how='left')
    fgi = aux['fng_value']
    return fgi


################################# NUEVAS FEATURES TRAIN
train['close_lag'] = train['close'].shift(1)
train['RSI'] = relative_strength_idx(train).fillna(0)
train['fgi'] = feargreedindex(train, feardata)
train = train.fillna(0)
print('SE MUESTRA EL TRAIN: ')
print(train)

################################## NUEVAS FEATURES TEST
# all the same for the test data
test['close_lag'] = test['close'].shift(1)
test['RSI'] = relative_strength_idx(test).fillna(0)
test['fgi'] = feargreedindex(test, feardata)
test = test.fillna(0)

#########################################
# Se fraccionan los datos de train en: train + validación
fraccion_train = 0.7  # Fracción de datos usada para entrenar
fraccion_valid = 1.00 - fraccion_train
train_aleatorio = train.sample(frac=1)
train = train_aleatorio.iloc[:int(fraccion_train * len(train)), :]
validacion = train_aleatorio.iloc[int(fraccion_train * len(train)):, :]

################# Se separa en features y target
train_X = train[columns]
train_y = train['TARGET']
valid_X = validacion[columns]
valid_y = validacion['TARGET']

# MODEL TRAINING
###################### MODELO LGBM ######################
folds = GroupKFold(n_splits=5)
params = {'objective': 'binary',
          'learning_rate': 0.02,
          "boosting_type": "gbdt",
          "metric": 'precision',
          'n_jobs': -1,
          'min_data_in_leaf': 32,
          'num_leaves': 1024,
          }
for fold_n, (train_index, valid_index) in enumerate(folds.split(train_X, train_y, groups=train['company'])):
    print(f'Fold {fold_n} started at {time.ctime()}')
    X_train, X_valid = train_X[columns].iloc[train_index], train_X[columns].iloc[valid_index]
    y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

    model = lgb.LGBMClassifier(**params, n_estimators=50)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)])
#####################################################

#################### SE DIBUJAN LAS FEATURES POR IMPORTANCIA #################
feature_importance = pd.DataFrame()
fold_importance = pd.DataFrame()
fold_importance["feature"] = columns
fold_importance["importance"] = model.feature_importances_
fold_importance["fold"] = fold_n + 1
feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

feature_importance["importance"] /= 5
# Se pintan las primeras 50 features
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features (avg over folds)');
# plt.show(block=False)
# plt.pause(5)
plt.savefig(pathOutput + "BOLSA_feature_importances.png")
# plt.close()
###################

##################### VALIDACIÓN ###################
print("COMIENZO DE VALIDACIÓN")
score = metrics.mean_absolute_error(valid_y, model.predict(valid_X))
print('CV score: {0:.4f}.'.format(score))
print("FIN DE VALIDACIÓN")
###############################################


# PREDICCIÓN independiente

# Eliminación de filas a null
test = filter_rows_by_values(test, "TARGET", ["null"])

submission = test[columns]
solucion = test['TARGET']

# Sólo aplicable si hay sigmoid en la última capa
prediccion = (model.predict(submission) > 0.5).astype("int32")
submission['TARGET']=prediccion

# Se añade la empresa en la primera columna y se guarda en Excel
submission=submission.join(test['company'])
submission.set_index(submission.pop('company'), inplace=True)
submission.reset_index(inplace=True)
submission.to_csv(os.path.join(pathOutput, 'Bolsa_DL_submission.csv'), index=False)

# PRECISION independiente
a=solucion
b=prediccion
TP=sum(1 for x,y in zip(a,b) if (x == y and y == 1))
TPandFP=sum(b)
precision= TP / TPandFP
print("TP: ", TP)
print("TP + FP: ", TPandFP)
print("---->>>>>> PRECISION (TP/(TP+FP)) FOR TEST DATASET: {0:.2f}% <<<<<<------".format(precision * 100))

# Tasa de desbalanceo
ift_mayoritaria = test[test.TARGET == False]
ift_minoritaria = test[test.TARGET == True]
tasaDesbalanceo = round(ift_mayoritaria.shape[0] / ift_minoritaria.shape[0], 2)
print("Tasa de desbalanceo = " + str(ift_mayoritaria.shape[0]) + "/" + str(
        ift_minoritaria.shape[0]) + " = " + str(tasaDesbalanceo))

print("Tasa de mejora de precisión respecto a random: ",
              round(precision / (1/(1+tasaDesbalanceo)), 2))

from sklearn.metrics import classification_report
print("Informe de metricas:")
print(classification_report(a, b))

# Rentabilidades
print("RENTABILIDADES. Se miran una a una, sin acumular:")

rentacompleta=test['renta_20dias']
valorrentabase=sum(rentacompleta)/len(rentacompleta)
print("Rentabilidad media del conjunto de test completo, sin filtrar: ", valorrentabase)

# Selección de renta de sólo elementos donde invertiría
a=test['renta_20dias']
b=prediccion
probabilidadessumadasainvertir=sum(x for x,y in zip(a,b) if (y == 1))
numeroinversiones=len(b)
valorrentaainvertir= probabilidadessumadasainvertir / numeroinversiones

print("Rentabilidad media del conjunto de test donde invertir, con filtro: ", valorrentaainvertir)


# Renta de la solución de TRAINING
b=train
b=b.loc[b['TARGET'] == 1]
probabilidadessumadas=sum(b['renta_20dias'])
numeroinversiones=len(b)
rentasolucion= probabilidadessumadas / numeroinversiones

print("Rentabilidad media del conjunto de la solución (TRAIN): ", rentasolucion)


print("END")