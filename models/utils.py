import torch
from loguru import logger
from sklearn.metrics import roc_auc_score



# ------------------- anomaly scores -------------------
def compute_anomaly_scores_graph_level(model, loader):
    model.eval()
    scores = []

    with torch.no_grad():
        for data in loader:
            data = data.to(model.device)
            z = model(data)
            scores.append(model.anomaly_score(z).cpu())

    return torch.cat(scores, dim=0)


def compute_anomaly_scores_node_level(model, data, mask_ind):
    model.eval()
    
    with torch.no_grad():
        data = data.to(model.device)
        z = model(data)
        scores = model.anomaly_score(z[mask_ind]).cuda()

    return scores

# ------------------- ratiooos -------------------

def get_ratios(pred, y, test_scores, compute_auc_roc=True):
    result = {}
    # test rate
    compare = pred == y
    n = len(compare)
    test_rate = compare.sum().item() / n
    result["test_rate"] = test_rate
    
    # confusion matrix... hehe
    true_positive = ((pred == 1) & (y == 1)).sum().item()
    true_negative = ((pred == 0) & (y == 0)).sum().item()
    false_positive = ((pred == 1) & (y == 0)).sum().item()
    false_negative = ((pred == 0) & (y == 1)).sum().item()
    
    result["true_positive"] = true_positive
    result["true_negative"] = true_negative
    result["false_positive"] = false_positive
    result["false_negative"] = false_negative
    
    # bunch of ther metrics
    eps = 1e-8 # dont wanna devide by zero
    TPR = true_positive / (true_positive + false_negative + eps)
    TNR = true_negative / (true_negative + false_positive + eps)
    precision = true_positive / (true_positive + false_positive + eps)
    balanced_accuracy = (TPR + TNR) / 2
    FPR = false_positive / (false_positive + true_negative + eps)

    result["recall"] = TPR
    result["specificity"] = TNR
    result["precision"] = precision
    result["balanced_accuracy"] = balanced_accuracy
    result["FPR"] = FPR
    if compute_auc_roc:
        result["ROC_AUC"] = roc_auc_score(y, test_scores)

    return result


def get_ratios_nl(data, test_mask, test_scores, R):
    pred = (test_scores > R).int()
    y = (data.y[test_mask] > 0).int()
    
    return get_ratios(pred, y, test_scores)


def get_ratios_gl(test_loader, test_scores, R):
    pred = (test_scores > R).int()
    y = []
    for data in test_loader:
        y.append(data.y)
    
    y_vec = torch.cat(y, dim=0)
    
    return get_ratios(pred, y_vec, test_scores)


# ------------------- decision boundary -------------------

def get_decision_boundary(y, test_scores):
    thresholds = torch.linspace(test_scores.min(), test_scores.max(), 500)

    best_R = None
    best_J = -1

    for R in thresholds:
        pred = (test_scores > R).int()
        res = get_ratios(pred, y, test_scores, compute_auc_roc=False)
        J = res["recall"] - res["FPR"] # youdens y

        if J > best_J:
            best_J = J
            best_R = R

    return best_R


def get_decision_boundary_nl(test_scores, data, test_mask):
    y = (data.y[test_mask] > 0).int()
    return get_decision_boundary(y, test_scores)


def get_decision_boundary_gl(test_loader, test_scores):
    y = []
    for data in test_loader:
        y.append(data.y)
    
    y_vec = torch.cat(y, dim=0)
    return get_decision_boundary(y_vec, test_scores)