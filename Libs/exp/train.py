import torch.nn.functional as F
import numpy as np


def train(model, optimizer, data_loader):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(model.device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
    return total_loss / len(data_loader.dataset)


def test(model, data_loader):
    model.eval()
    accs = []
    for data in data_loader:
        data = data.to(model.device)
        logits = model(data.x, data.edge_index)
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return np.mean(accs)
