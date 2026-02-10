import torch
from loguru import logger

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
        scores = model.anomaly_score(z[mask_ind]).cpu()

    return scores


def get_test_rate_nl(data, test_mask, test_scores, R):
    pred = (test_scores > R).int()
    
    # all labels except of 0 is an anomaly
    y = (data.y[test_mask] > 0).int()
    compare = pred == y

    test_rate = compare.sum().item() / len(compare)
    
    logger.info(test_rate)
    logger.info(f"right classified: {compare.sum().item()} out of {len(compare)}")

    return test_rate


def get_test_rate_gl(test_loader, test_scores, R):
    pred = (test_scores > R).int()
    y = []
    for data in test_loader:
        y.append(data.y)
    
    y_vec = torch.cat(y, dim=0)
    compare = pred == y_vec

    right_classified = compare.sum().item()
    test_rate = right_classified / len(compare)
    logger.info(test_rate)
    logger.info(f"right classified: {right_classified} out of {len(compare)}")
    
    return test_rate