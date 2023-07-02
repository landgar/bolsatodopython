import lightgbm as lgb
from imblearn.under_sampling import *
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_validate, StratifiedShuffleSplit
from sklearn import metrics
from sklearn.svm import LinearSVC

from SplitterInformacion import *
from CSV import *
import time
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.combine import *
from sklearn.model_selection import train_test_split


def generaModeloLightGBM(datos, metrica, pintarFeatures=False, pathCompletoDibujoFeatures=""):
    # Aleatorización de datos
    df = aleatorizarDatos(datos)

    # Se eliminan datos no numéricos
    df = clean_dataset(df)

    # Se trocean los datos para entrenar y validar
    X_train, X_test, y_train, y_test = divideDatosParaTrainTestXY(df)

    # Imbalanced data: se transforma el dataset
    # transform the dataset
    smote = SMOTEENN(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    params = {'objective': 'binary',
              'learning_rate': 0.02,
              "boosting_type": "gbdt",
              "metric": metrica,
              'n_jobs': -1,
              'min_data_in_leaf': 32,
              'num_leaves': 1024,
              }
    model = lgb.LGBMClassifier(**params, n_estimators=50)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)])

    # DEBUG:
    y_pred_train = model.predict(X_train)
    print("SE DEBE OPTIMIZAR LA PRECISIÓN Y EL RECALL. Esto es con el train, asi que debería salir 100%:")
    print('Training-set LGB precision score: {0:0.4f}'.format(precision_score(y_train, y_pred_train)))
    print('Training-set LGB recall score: {0:0.4f}'.format(recall_score(y_train, y_pred_train)))

    # DEBUG:
    y_pred_test = model.predict(X_test)
    print("Para los datos de VALIDACIÓN, básico, sin proba:")
    print('Precision score: {0:0.4f}'.format(precision_score(y_test, y_pred_test)))
    print('Recall score: {0:0.4f}'.format(recall_score(y_test, y_pred_test)))

    # DEBUG:
    if pintarFeatures:
        #################### SE DIBUJAN LAS FEATURES POR IMPORTANCIA #################
        columns = [col for col in X_train.columns]
        feature_importance = pd.DataFrame()
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = columns
        fold_importance["importance"] = model.feature_importances_
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        # Se pintan las primeras x features
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:10].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
        features_ordenadas=best_features.sort_values(by="importance", ascending=False)
        print("Features por importancia:")
        print(features_ordenadas)
        plt.figure(figsize=(16, 12));
        sns.barplot(x="importance", y="feature", data=features_ordenadas);
        plt.title('Importance of Feature');
        # plt.show(block=False)
        # plt.pause(5)
        plt.savefig(pathCompletoDibujoFeatures)
        # plt.close()
        ###################

    # DEBUG:
    # visualize confusion matrix with seaborn heatmap
    cm = confusion_matrix(y_test, y_pred_test)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

    # DEBUG:
    calculaPrecision(y_test, y_pred_test)

    # DEBUG:
    umbral = 0.7
    pred, proba = predictorConProba(model, X_test, umbralProba=umbral, analizarResultado=True,
                                    y_solucionParaAnalisis=y_test,
                                    mensajeDebug="Análisis en la creación el modelo mirando la probabilidad de TARGET==1 y filtrando por proba: " + str(
                                        umbral))

    return model


def calculaPrecision(target, prediccion):
    # PRECISION (TP/(TP+FP))
    TP = sum(1 for x, y in zip(target, prediccion) if (x == y and y == 1))
    TPandFP = sum(prediccion)
    precision = TP / TPandFP

    # DEBUG
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1
    print("TP: ", TP)
    print("TP + FP: ", TPandFP)
    print("PRECISION: ", precision)
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(target, prediccion, digits=3))

    return precision


def evaluate(model, testing_set_x, testing_set_y):
    predictions = model.predict_proba(testing_set_x)

    accuracy = accuracy_score(testing_set_y, predictions[:, 1] >= 0.5)
    roc_auc = roc_auc_score(testing_set_y, predictions[:, 1])
    precision = precision_score(testing_set_y, predictions[:, 1] >= 0.5)
    recall = recall_score(testing_set_y, predictions[:, 1] >= 0.5)
    pr_auc = average_precision_score(testing_set_y, predictions[:, 1])

    result = pd.DataFrame([[accuracy, precision, recall, roc_auc, pr_auc]],
                          columns=['Accuracy', 'Precision', 'Recall', 'ROC_auc', 'PR_auc'])
    return (result)


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)

    # Se quitan las siguientes columnas, sólo si existen:
    if set(['date', 'ticker']).issubset(df.columns):
        df = df.drop(columns=['date', 'ticker'])

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)


def divideDatosParaTrainTestXY(datos):
    # Se separan features de TARGET, y se quitan columnas no numéricas
    X, y = limpiaDatosParaUsarModelo(datos)
    X_A, X_B, y_A, y_B = train_test_split(X, y, stratify=y, test_size=0.3, random_state=5)
    return X_A, X_B, y_A, y_B


def limpiaDatosParaUsarModelo(datos):
    # Se separan features de TARGET, y se quitan columnas no numéricas
    # Se devuelven las columnas de features limpias y target
    X = datos.drop(columns=['TARGET'])
    X = X.drop(columns=['INCREMENTO'])
    X = X.drop(columns=['diff'])
    y = datos['TARGET']
    return X, y


def troceaDataframeMismoTamano(datos, numPartesIguales):
    # Se reparten las filas entre los dataframes indicados. No se hace aleatorización previa.
    # Se revuelven tantos dataframes del mismo tamaño como se indique como parámetro
    return np.array_split(datos, numPartesIguales)


def predictorConProba(modelo, X, umbralProba=0.8, analizarResultado=False, y_solucionParaAnalisis=pd.DataFrame(),
                      mensajeDebug=""):
    # Se intenta subir la precisión en base a coger sólo las predicciones de alta probabilidad
    # Parámetros:
    # Modelo: modelo ya entrenado
    # X: features limpias
    # y_validacion: Opcionalmente
    # Se devuelve la predicción y su probabilidad

    if analizarResultado == True:
        aux = pd.concat([X, y_solucionParaAnalisis], axis=1)
        aux = clean_dataset(aux)
        auxSinTarget = aux.drop(columns=['TARGET'])
    else:
        aux = X
        aux = clean_dataset(aux)
        auxSinTarget = aux

    # Se analizan sólo las predicciones a 1
    aux["proba"] = modelo.predict_proba(auxSinTarget)[:, 1]
    threshold = umbralProba
    aux["pred"] = aux["proba"].apply(lambda el: 1.0 if el >= threshold else 0.0)

    print("umbralProba: ", umbralProba)

    if analizarResultado == True:
        print(mensajeDebug)
        print("MODIFICAR EL UMBRAL de 0.5 LLEVA A OVERFITTING")
        auxSoloTarget = aux['TARGET']
        print('Precision score: {0:0.4f}'.format(precision_score(auxSoloTarget, aux["pred"])))
        print('Recall score: {0:0.4f}'.format(recall_score(auxSoloTarget, aux["pred"])))
        calculaPrecision(auxSoloTarget, aux["pred"])

    return aux["pred"], aux["proba"]
