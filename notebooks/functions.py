from sklearn.metrics import roc_auc_score

def gini_coeficient(y_test, y_pred_probs):

    area_under_reciver = roc_auc_score(y_test, y_pred_probs)
    return 2 * area_under_reciver - 1

