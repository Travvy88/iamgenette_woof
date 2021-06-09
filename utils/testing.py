import torch
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm

def test_classic(net, val_loader, device):
    """
    Test network on val data
    """
    net.eval()
    with torch.no_grad():

        preds, trues = [], []

        for X, y in tqdm(val_loader):
            X = X.to(device)
            y = y.to(device)

            y_pred = net(X)
            y_pred = y_pred.max(1)[1]
            
            preds.append(y_pred)
            trues.append(y)

        preds = torch.cat(preds).detach().cpu().numpy()  
        trues = torch.cat(trues).detach().cpu().numpy()
        
    print(
        '\nAccuracy =', accuracy_score(preds, trues),
        '\nF-score =', f1_score(trues, preds, average='weighted'),
        '\n'
    )
            