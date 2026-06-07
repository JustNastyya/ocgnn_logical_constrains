import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from torchlogix.layers import LogicDense, FixedBinarization, GroupSum
from sklearn.feature_selection import SelectKBest, mutual_info_classif



GATE_CODES = {
    2: "INPUT", 3: "NOT_INPUT",
    4: "AND", 6: "OR",
}


_SIMPLE_POSITIVE = {2, 4}
_SIMPLE_NEGATED = {3}

LUT_RANK = 2
TOP_K_FEATURES = 20
EPOCHS = 500


def train_logix_model(X_cat, y, config):
    # selecting top 20 features
    
    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=TOP_K_FEATURES
    )

    X_selected = selector.fit_transform(X_cat, y)
    X_selected = torch.tensor(
        X_selected,
        dtype=torch.float32
    )
    y = torch.tensor(y, dtype=torch.long)
    
    # start debug
    X_selected = X_selected[:20]
    y = y[:20]
    # end debug

    logger.info(X_selected.shape)
    
    n, d = X_selected.shape
    epochs = EPOCHS
    lr = config["lr"]
    

    model = nn.Sequential(
        LogicDense(20, 16, lut_rank=2),
        LogicDense(16, 12, lut_rank=2),
        LogicDense(12, 6, lut_rank=2),
        LogicDense(6, 4, lut_rank=2),
        LogicDense(4, 2, lut_rank=2)
        
        # GroupSum(k=2, tau=4),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_selected)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            pred = output.argmax(dim=1)
            acc = (pred == y).float().mean()
            logger.info(f"Epoch {epoch+1:03d} | loss={loss.item():.4f} | acc={acc:.4f}")
    
    # start: debug
    with torch.no_grad():
        output = model(X_selected)
        pred = output.argmax(dim=1)

    print("labels:", y)
    print("preds :", pred)

    print("label counts:", torch.bincount(y))
    print("pred counts :", torch.bincount(pred))
    # end: debug
    return model


def _resolve_gate_types(layer):
    if hasattr(layer, "gate_types"):
        logits = layer.gate_types
        if logits.dim() == 3:
            return logits.argmax(dim=-1)
        return logits
    return None


def _resolve_connections(layer):
    if hasattr(layer, "connections"):
        conn = layer.connections
        if hasattr(conn, "indices"):
            return conn.indices
        return conn
    return None


def _gate_to_predicted_class(gate_code):
    if gate_code in _SIMPLE_POSITIVE:
        return 1
    if gate_code in _SIMPLE_NEGATED:
        return 0
    return None


def _gate_to_condition(gate_code, feature_index):
    if gate_code in (2, 4):
        return {"feature_index": feature_index, "op": ">", "threshold": 0.5}
    if gate_code == 3:
        return {"feature_index": feature_index, "op": "<=", "threshold": 0.5}
    return None


def _is_or_gate(gate_code):
    return gate_code == 6


def extract_logix_rules(model, cat_indices):
    logic_layers = [m for m in model.modules() if isinstance(m, LogicDense)]

    if len(logic_layers) < 2:
        logger.warning("Fewer than 2 LogicDense layers found, skipping rule extraction")
        return []

    layer1 = logic_layers[0]
    layer2 = logic_layers[1]

    gates1 = _resolve_gate_types(layer1)
    gates2 = _resolve_gate_types(layer2)
    conn1 = _resolve_connections(layer1)
    conn2 = _resolve_connections(layer2)

    if conn2 is None:
        logger.warning("Could not resolve connections from layer 2, skipping logix extraction")
        return []

    if gates2 is None:
        logger.warning("Could not resolve gate types from layer 2, skipping logix extraction")
        return []

    # Determine lut_rank from available data
    lut_rank_2 = conn2.shape[1] if conn2.dim() == 2 else gates2.shape[1]

    if cat_indices is None:
        cat_indices = list(range(gates1.shape[1] if gates1 is not None else 0))

    rules = []

    for slot in range(lut_rank_2):
        hidden_idx = int(conn2[1, slot].item()) if conn2 is not None and conn2.shape[0] > 1 else slot
        gate_code = int(gates2[1, slot].item()) if gates2 is not None and gates2.shape[0] > 1 else 2

        if hidden_idx < 0 or hidden_idx >= (gates1.shape[0] if gates1 is not None else 0):
            continue

        predicted_class = _gate_to_predicted_class(gate_code)
        if predicted_class is None:
            continue

        if conn1 is None:
            continue

        lut_rank_1 = conn1.shape[1] if conn1.dim() == 2 else (gates1.shape[1] if gates1 is not None else 0)

        and_conditions = []
        or_conditions = []

        for slot_h in range(lut_rank_1):
            feat_idx = int(conn1[hidden_idx, slot_h].item())
            if feat_idx < 0:
                continue

            uni_idx = int(cat_indices[feat_idx]) if feat_idx < len(cat_indices) else feat_idx
            feat_gate = int(gates1[hidden_idx, slot_h].item()) if gates1 is not None else 2

            if _is_or_gate(feat_gate):
                cond = _gate_to_condition(2, uni_idx)
                if cond is not None:
                    or_conditions.append(cond)
            else:
                cond = _gate_to_condition(feat_gate, uni_idx)
                if cond is not None:
                    and_conditions.append(cond)

        if and_conditions:
            rules.append({
                "conditions": and_conditions,
                "predicted_class": predicted_class,
                "meta": {"source": "logix", "gate_type": "AND"},
            })

        for cond in or_conditions:
            rules.append({
                "conditions": [cond],
                "predicted_class": predicted_class,
                "meta": {"source": "logix", "gate_type": "OR"},
            })

    logger.info(f"Extracted {len(rules)} rules from logic gate model")
    return rules
