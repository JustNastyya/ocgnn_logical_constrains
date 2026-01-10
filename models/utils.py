import torch

def compute_anomaly_scores(model, loader):
    model.eval()
    scores = []

    with torch.no_grad():
        for data in loader:
            data = data.to(model.device)
            z = model(data)
            scores.append(model.anomaly_score(z).cpu())

    return torch.cat(scores, dim=0)
