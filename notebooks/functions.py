from sklearn.metrics import roc_auc_score

def gini_coeficient(model, X_test, Y_Test):
    """
    Cálculo do Coeficiente de Gini
    """

    # Obtenção de Probabilidades (importante para o Gini)
    # retorna a chance do cliente ser 1 (Attrited Customer)
    probs = model.predict_proba(X_test)[:, 1]

    area_under_reciver = roc_auc_score(Y_Test, probs)
    print(f"Area Under Reciver: {area_under_reciver}")

    return (2 * area_under_reciver) - 1
