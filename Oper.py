import torch
import torch.nn.functional as F
import Loss as L

import numpy as np
from skimage.filters import threshold_otsu
from sklearn.metrics import jaccard_score

def train_model(model, loader, device, optim, loss_func, pad, gamma, alpha):
    model.train()
    train_loss = 0
    train_jaccard = 0
    
    for idx, (X, y) in enumerate(loader):
        optim.zero_grad()
        X = X.to(device)# [N, 1, H, W]
        y = y.to(device)  
        pred = model(X)# [N, 2, H, W]

        y = torch.squeeze(y,1).long()# [N, H, W] with class indices (0, 1)
        pred = pred[:, :, pad:-pad, pad:-pad]#裁剪预测结果
        pred_act = (torch.sigmoid(pred).squeeze())[0]#利用sigmoid激活预测结果
        if (torch.median(pred_act) > torch.mean(pred_act)):#选择背景为0的通道
            pred_act = 1 - pred_act
        #获得二值化jaccard作为粗略估计
        pred_vis = pred_act.detach().cpu().numpy()
        threshold = threshold_otsu(pred_vis)#计算Otsu二值化
        pred_bin = np.where(pred_vis > threshold, 1, 0).astype(np.uint8)
        y_vis = (y.detach().cpu().squeeze().numpy()).astype(np.uint8)
        train_jaccard += jaccard_score(y_vis.ravel(), pred_bin.ravel())
        
        if loss_func == 'focal_loss':#gamma = 0 为BCE Loss
            loss = L.focal_loss(gamma=gamma, alpha=alpha)(pred, y)
        elif loss_func == 'cross_entropy':
            loss = F.cross_entropy(pred, y)
        elif loss_func == 'dice_loss':
            loss = L.dice_loss(pred_act, y)
        else:
            raise ValueError('No such loss function. Only focal_loss, dice_loss and cross_entropy are available.')

        train_loss += loss.cpu().detach().numpy()
        
        with torch.autograd.detect_anomaly():
            loss.backward()
        optim.step()

        del X, y, pred, pred_act, loss
        torch.cuda.empty_cache()
    
    return train_jaccard, train_loss

def val_model(model, loader, device, loss_func, pad, gamma, alpha):
    model.eval()
    val_loss = 0
    val_jaccard = 0
    
    with torch.no_grad():
        for idx, (X, y) in enumerate(loader):
            X = X.to(device)# [N, 1, H, W]
            y = y.to(device)  
            pred = model(X)# [N, 2, H, W]

            y = torch.squeeze(y, 1).long()# [N, H, W] with class indices (0, 1)
            pred = pred[:, :, pad:-pad, pad:-pad]#裁剪预测结果
            pred_act = (torch.sigmoid(pred).squeeze())[0]#利用sigmoid激活预测结果
            if (torch.median(pred_act) > torch.mean(pred_act)):#选择背景为0的通道
                pred_act = 1 - pred_act
            #获得二值化jaccard作为粗略估计
            pred_vis = pred_act.detach().cpu().numpy()
            threshold = threshold_otsu(pred_vis)
            pred_bin = np.where(pred_vis > threshold, 1, 0).astype(np.uint8)
            y_vis = (y.detach().cpu().squeeze().numpy()).astype(np.uint8)
            val_jaccard += jaccard_score(y_vis.ravel(), pred_bin.ravel())

            if loss_func == 'focal_loss':#gamma = 0 为BCE Loss
                loss = L.focal_loss(gamma=gamma, alpha=alpha)(pred, y)
            elif loss_func == 'cross_entropy':
                loss = F.cross_entropy(pred, y)
            elif loss_func == 'dice_loss':
                loss = L.dice_loss(pred_act, y)
            else:
                raise ValueError('No such loss function. Only focal_loss, dice_loss and cross_entropy are available.')

            val_loss += loss.cpu().detach().numpy()
            
            del X, y, pred, pred_act, loss
            torch.cuda.empty_cache()
    
    return val_jaccard, val_loss

def run_model(model, optim, train_loader, val_loader, device,
                save_path,
                epochs,
                train_size,
                val_size,
                pad,
                loss_func,
                lr, 
                betas, 
                eps, 
                weight_decay, 
                gamma, 
                alpha):

    torch.autograd.set_detect_anomaly(True)
    
    all_train_losses = []
    all_train_jaccards = []
    all_val_losses = []
    all_val_jaccards = []
    
    for epoch in range(epochs):
        print('epoch:',epoch+1)
        
        train_jaccard, train_loss = train_model(model, train_loader, device, optim=optim, 
                                 loss_func=loss_func, pad=pad, gamma=gamma, alpha=alpha)
        val_jaccard, val_loss = val_model(model, val_loader, device, 
                             loss_func=loss_func, pad=pad, gamma=gamma, alpha=alpha)
        
        train_jaccard /= train_size
        train_loss /= train_size
        val_jaccard /= val_size
        val_loss /= val_size
        #保存最优结果
        if val_loss < min(all_val_losses + [256]):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'loss': train_loss,
                }, save_path)
        
        all_train_losses.append(train_loss)
        all_train_jaccards.append(train_jaccard)
        all_val_losses.append(val_loss)
        all_val_jaccards.append(val_jaccard)
    
    return (all_train_losses, all_train_jaccards, all_val_losses, all_val_jaccards)