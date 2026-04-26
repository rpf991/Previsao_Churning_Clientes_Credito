from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gini_coeficient(model, X_test, Y_Test, classe=1):
    """
    Cálculo do Coeficiente de Gini
    """

    # Obtenção de Probabilidades (importante para o Gini)
    # retorna a chance do cliente ser 'Attrited Customer'
    y_probs = model.predict_proba(X_test)[:, classe]

    area_under_reciver = roc_auc_score(Y_Test, y_probs)
    print(f"Area Under Reciver: {area_under_reciver}")

    return (2 * area_under_reciver) - 1


def lorenz_curve(Y_test, Y_prob):
    """
    Definição da Curva de Lorenz
    """

    df_lorenz = pd.DataFrame({"real": Y_test, "prob": Y_prob}).sort_values(by='prob', ascending=False)

    df_lorenz['cum_churn'] = df_lorenz['real'].cumsum()/df_lorenz['real'].sum()
    df_lorenz['perc'] = np.arange(1, len(df_lorenz) + 1)/len(df_lorenz)

    plt.plot(df_lorenz['perc'], df_lorenz['cum_churn'], label='Curva de Lorenz')
    plt.xlabel("Percentual da População")
    plt.ylabel("Percentual Acumulado de Churn")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Linha perfeita de Igualdade')
    plt.grid(True)
    plt.legend()
    plt.title("Gráfico de Lorenz")
    plt.show()

    