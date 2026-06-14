from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gini_coeficient(model, X_test, Y_Test):
    """
    Cálculo do Coeficiente de Gini
    """
    # Obtenção de Probabilidades (importante para o Gini)
    # retorna a chance do cliente ser 'Attrited Customer'
    y_probs = model.predict_proba(X_test)[:, 1]

    area_under_reciver = roc_auc_score(Y_Test, y_probs)
    #print(f"Area Under Reciver: {area_under_reciver}")
    
    gini = 2 * area_under_reciver - 1

    return area_under_reciver, gini


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


def gmean_max(model, X_test, Y_test):
    """
    O G-men avalia o equilibrio entre quem vai sair e quem vai entrar 

    TRP -> True Positive Rate (mede a proporção entre positivos reais que foram previstos corretamente no modelo)
    FRP -> False Positive Rate (mede a proporção entre negativos reais que foram previstos corretamente no modelo)
    """
    # 1) Obtenção das probabilidades da classe 1
    y_probs = model.predict_proba(X_test)[:, 1]

    # 2) Calculo da curva de ROC (grafico usado para avaliar os desempenhos dos modelos de classificação binária)
    frp, trp, thresholds = roc_curve(Y_test, y_probs)

    # 3) Calculo do G-Mean para cada threshold
    gmeans = np.sqrt(trp * (1-frp))

    # 4) Encontrar o indice do maior gmean
    idx = np.argmax(gmeans)
    limite_ideal = thresholds[idx]

    print(f'Melhor G-Means: {gmeans[idx]}')

    return limite_ideal
