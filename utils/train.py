import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import numpy as np

def classic_train(net, freq, device, loss_fn, optimizer, 
                  train_loader, val_loader, num_epoch, NUM_TEST, NAME_TEST, begin=0):
    """
    Train network
    
    Args:
    -- net: PyTorch model
    -- freq: frequency of validation 
    (e.g. freq = 100 means every 100 iterations it will validate)
    -- device: 'cuda' or 'cpu'
    -- loss_fn: function that calculate loss function
    -- optimizer: optimizer from torch.nn.optim\
    -- train_loader: torch.utils.data.DataLoader with train data
    -- val_loader: torch.utils.data.DataLoader with validation data
    -- num_epoch: how many epochs should it train
    Args for save weigths:
    -- NUM_TEST: number of attempt in experimenat
    -- NAME_TEST: name of experiment
    -- begin: from which epoch begin training
    
    Out: history of each epoch: train loss, val loss, accuracy, f1 score
    """
    train_losses = []
    val_losses = []
    f1 = []
    accuracy = []
       
    
    itr = 0  
    for i in range(num_epoch):  
        epoch = i + begin
        print('-- EPOCH', epoch, '---------------------')
        for X, y in tqdm(train_loader):
            net.train()
            X = X.to(device)
            y = y.to(device)
            y_pred = net(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())   
            
            
            if itr % freq == 0:
                print('Validating...')
                net.eval()
                with torch.no_grad():
                    now_val_losses = []
                    preds, trues = [], []

                    for X, y in val_loader:
                        X = X.to(device)
                        y = y.to(device)
                        
                        y_pred = net(X)
                        loss = loss_fn(y_pred, y)
                        now_val_losses.append(loss.item())
                        
                        y_pred = y_pred.max(1)[1]
                        preds.append(y_pred)
                        trues.append(y)
                        
                    preds = torch.cat(preds).detach().cpu().numpy()  
                    trues = torch.cat(trues).detach().cpu().numpy()
                    
                    
                    val_losses.append(np.mean(now_val_losses).item()) 
                    accuracy.append(accuracy_score(trues, preds))
                    f1.append(f1_score(trues, preds, average='weighted'))
                    
                    torch.save(net.state_dict(), f'weights/{NAME_TEST}/test-{NUM_TEST}-epoch-{epoch}-iter-{itr}.pth')  # сохраняем веса эпох

                print(
                    'Epoch', epoch,
                    'Iteration', itr,
                    '\nTrain loss =', np.mean(train_losses[itr - freq:itr]),
                    '\nVal loss =', val_losses[-1],
                    '\nAccuracy =', accuracy[-1] ,
                    '\nF-score =', f1[-1],
                    '\n'
                    
                )
            
            itr += 1
        
    return train_losses, val_losses, accuracy, f1


def alphaseg_train(net, segnet, freq, device, loss_fn, optimizer, 
                       train_loader, val_loader, num_epoch, NUM_TEST, NAME_TEST, begin=0):
    """
    In development
    """
    train_losses = []
    val_losses = []
    f1 = []
    accuracy = []
    segnet.eval()
    
    itr = 0  # счетчик итераций
    for i in range(num_epoch):  
        epoch = i + begin
        print('-- EPOCH', epoch, '---------------------')
        for X, y in tqdm(train_loader):
            net.train()
            X = X.to(device)
            y = y.to(device)
            
            seg = segnet(X)
            
            where = np.where(
                (predictions['labels'].cpu().detach().numpy() == 18) & 
                (predictions['scores'].cpu().detach().numpy() > 0.3)
            )[0]
            
            

            mask = []
            for i in where:
                mask.append(predictions['masks'].cpu().detach().numpy()[i])
            mask = np.concatenate(mask).sum(0)
            mask = np.clip(mask, 0, 1)
            
            y_pred = net(X)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())   
            
            
            if itr % freq == 0:
                print('Validating...')
                net.eval()
                with torch.no_grad():
                    now_val_losses = []
                    preds, trues = [], []

                    for X, y in val_loader:
                        X = X.to(device)
                        y = y.to(device)
                        
                        y_pred = net(X)
                        loss = loss_fn(y_pred, y)
                        now_val_losses.append(loss.item())
                        
                        y_pred = y_pred.max(1)[1]
                        preds.append(y_pred)
                        trues.append(y)
                        
                    preds = torch.cat(preds).detach().cpu().numpy()  
                    trues = torch.cat(trues).detach().cpu().numpy()
                    
                    
                    val_losses.append(np.mean(now_val_losses).item()) 
                    accuracy.append(accuracy_score(trues, preds))
                    f1.append(f1_score(trues, preds, average='weighted'))
                    
                    torch.save(net.state_dict(), f'weights/{NAME_TEST}/test-{NUM_TEST}-epoch-{epoch}-iter-{itr}.pth')  # сохраняем веса эпох

                print(
                    'Epoch', epoch,
                    'Iteration', itr,
                    '\nTrain loss =', np.mean(train_losses[itr - freq:itr]),
                    '\nVal loss =', val_losses[-1],
                    '\nAccuracy =', accuracy[-1] ,
                    '\nF-score =', f1[-1],
                    '\n'
                    
                )
            
            itr += 1
        
    return train_losses, val_losses, accuracy, f1
