import gc
import torch
import numpy as np
from MIL_dataloader import MIL_dataloader
from tqdm import tqdm
from model import HACL_Net
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import os
from utils.Early_Stopping import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import argparse
from collections import deque
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    

parser = argparse.ArgumentParser(description='HACL-Net for PAS Diagnosis')
parser.add_argument('--feat_path', type=str, default='/media/data0/placenta/placenta_processed_data/patient/new_all_patches_npz',
                    help='MRI Slices of each patient (e.g. npz files)')
parser.add_argument('--img_label_path', type=str, default='/media/data0/placenta/placenta_processed_data/patient/SAG.csv')
parser.add_argument('--nepochs', type=int, default=40, help='The maxium number of epochs to train')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--l1_reg', default=0, type=float, help='l1 regulation factor (default: 1e-5)')
parser.add_argument('--debug', default=False, type=bool, help='debug mode (default: False)')


def prediction(model, queryloader, criterion, testing=False):
    model.eval()

    lbl_pred_all = None
    label_all = []
    iter = 0

    tbar = tqdm(queryloader, desc='\r')
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tbar):

            X, lbl = sampled_batch['feat'], sampled_batch['label']

            graph = X[0].cuda()
            lbl = lbl.cuda()
            label = lbl.data.cpu().numpy()
            label = np.squeeze(label)

            Y_pred = model(graph)
            one_max = -1.0
            index = None
            for i, l in enumerate(Y_pred):
                l = torch.nn.Softmax(dim=0)(l)
                if l[1].data >= one_max:
                    one_max = l[1].data
                    index = i

            lbl_pred = Y_pred[index,:].unsqueeze(0)
            label_all.append(label)

            if iter == 0:
                lbl_pred_all = lbl_pred
                lbl_torch = lbl
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_torch = torch.cat([lbl_torch, lbl])
            iter += 1

    label_all = np.asarray(label_all)

    # A loss function combined survive time and status; we main only need status (class label)
    label_all_tensor = torch.from_numpy(label_all).long()
    lbl_pred_tensor = lbl_pred_all.cpu()
    loss_surv = criterion(lbl_pred_tensor, label_all_tensor)
        
    loss_surv = criterion(lbl_pred_tensor.squeeze(1), label_all_tensor)

    l1_reg = None
    for W in model.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()  

    loss = loss_surv + args.l1_reg * l1_reg
    print("\nval_loss_nn: %.4f, L1: %.10f" % (loss_surv, args.l1_reg * l1_reg))

    # Metrics
    pred_all = lbl_pred_all.data.max(1)[1].type_as(lbl_torch) # argmax
    accur = accuracy_score(lbl_torch.cpu(), pred_all.cpu())
    f1 = f1_score(lbl_torch.cpu(), pred_all.cpu(), average='micro')
    confusion = confusion_matrix(lbl_torch.cpu(), pred_all.cpu())
    
    auc = roc_auc_score(lbl_torch.cpu(), pred_all.cpu())
    fpr, tpr, thresholds = roc_curve(lbl_torch.cpu(), pred_all.cpu())

    if not testing:
        print('\n[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
            'accuracy: {:.4f}'.format(accur), "F1-Score:{:.4f}".format(f1), "AUC-Score:{:.4f}".format(auc),
            'Confusion Matrix: {}'.format(confusion))
    else:
        print('\n[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
            'accuracy: {:.4f}'.format(accur), "F1-Score:{:.4f}".format(f1), "AUC-Score:{:.4f}".format(auc),
            'Confusion Matrix: {}'.format(confusion))

    return loss.data.item(), f1


def pick_one(Y_pred):
    Y_pred = torch.nn.Softmax(dim=1)(Y_pred)
    red = -1.0
    index = None

    for i, l in enumerate(Y_pred):
        if l[1].data >= red:
            red = l[1].data
            index = i
    return index


def train_epoch(epoch, model, optimizer, trainloader, criterion, verbose=1):
    model.train()

    lbl_pred_all = None
    lbl_pred_each = None
    lbl_torch = None

    label_all = []

    iter = 0
    gc.collect()
    loss_nn_all = []
    
    tbar = tqdm(trainloader, desc='\r')
    zero_queue = deque()
    one_queue = deque()
    for i_batch, sampled_batch in enumerate(tbar):
        X, lbl = sampled_batch['feat'], sampled_batch['label']

        X_pair = None
        lbl_pair = None
        target = None
        if int(lbl.data) == 0:
            if len(one_queue) == 0:
                zero_queue.append(X)
                continue
            else: 
                X_pair = one_queue.popleft()
                lbl_pair = torch.ones(1,1)
        elif int(lbl.data) == 1:
            if len(zero_queue) == 0: 
                one_queue.append(X)
                continue
            else: 
                X_pair = zero_queue.popleft()
                lbl_pair = torch.zeros(1,1)
        else: raise 0
        
        graph = X[0].cuda()
        graph_pair = X_pair[0].cuda()
        lbl = lbl.cuda()
        lbl_pair = lbl_pair.cuda()

        # Forward
        Y_pred = model(graph)
        Y_pred_pair = model(graph_pair)
        lbl_pred = None
        lbl_pred_pair = None
        lbl_pred = Y_pred[pick_one(Y_pred), :].unsqueeze(0)
        lbl_pred_pair = Y_pred_pair[pick_one(Y_pred_pair), :].unsqueeze(0)

        label = lbl.data.cpu().numpy()
        label = np.squeeze(label)
        label_pair = lbl_pair.data.cpu().numpy()
        label_pair = np.squeeze(label_pair)
        label_all.append(label)
        label_all.append(label_pair)
        
        if lbl_pred_all == None or lbl_torch == None:
            lbl_pred_all = lbl_pred
            lbl_pred_all = torch.cat([lbl_pred_all,lbl_pred_pair])
            lbl_torch = lbl
            lbl_torch = torch.cat([lbl_torch, lbl_pair])
        else:
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred, lbl_pred_pair])
            lbl_torch = torch.cat([lbl_torch, lbl, lbl_pair])

        if lbl_pred_each == None:
            lbl_pred_each = lbl_pred
            lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred_pair])
        else:
            lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred, lbl_pred_pair])
        
        # Loss Function in a batch
        iter += 1
        if iter % 1 == 0 or i_batch == len(trainloader)-1:
            label_all = np.asarray(label_all)
            optimizer.zero_grad()  # zero the gradient buffer

            label_all_tensor = torch.from_numpy(label_all).long()
            lbl_pred_each_cpu = lbl_pred_each.cpu()
            loss_surv = criterion(lbl_pred_each_cpu, label_all_tensor)

            # L1 Regulation
            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()

            # Contrastive Learning
            MRLoss = torch.nn.MarginRankingLoss(margin=0.5)
            target = torch.tensor([1.]).cuda()
            if int(lbl.data) == 1: loss_MR = MRLoss(torch.nn.Softmax(dim=0)(lbl_pred[0])[1].unsqueeze(0), torch.nn.Softmax(dim=0)(lbl_pred_pair[0])[1].unsqueeze(0), target)
            else: loss_MR = MRLoss(torch.nn.Softmax(dim=0)(lbl_pred_pair[0])[1].unsqueeze(0), torch.nn.Softmax(dim=0)(lbl_pred[0])[1].unsqueeze(0), target)

            loss = loss_surv + args.l1_reg * l1_reg + loss_MR

            # Backward
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            lbl_pred_each = None

            label_all = []
            loss_nn_all.append(loss.data.item())
            gc.collect()


    pred_all = lbl_pred_all.data.max(1)[1].type_as(lbl_torch)
    accur = accuracy_score(lbl_torch.cpu(), pred_all.cpu())
    f1 = f1_score(lbl_torch.cpu(), pred_all.cpu(), average='micro')
    confusion = confusion_matrix(lbl_torch.cpu(), pred_all.cpu())
    auc = roc_auc_score(lbl_torch.cpu(), pred_all.cpu())

    if verbose > 0:
        print("\nEpoch: {}, loss_nn: {}, MRLoss: {}".format(epoch, np.mean(loss_nn_all), loss_MR))
        print('\n[Training]\t loss (nn):{:.4f}'.format(np.mean(loss_nn_all)),
            'accuracy: {:.4f}'.format(accur), "F1-Score:{:.4f}".format(f1), "AUC-Score:{:.4f}".format(auc),
            'Confusion Matrix: {}'.format(confusion))


def train(train_path, test_path, model_save_path, label_train_val, num_epochs, lr):      
    model = HACL_Net().cuda()

    parameters_feature_net = list(model.feature_net.parameters())
    parameters_classfier = list(model.fc6.parameters())
    optimizer = torch.optim.Adam([{'params': parameters_feature_net, 'lr': lr/10},{'params': parameters_classfier, 'lr': lr*10}], 
                                  lr = lr, weight_decay = 0)

    loss_f = torch.nn.CrossEntropyLoss()

    Data = MIL_dataloader(data_path=train_path, label_train_val=label_train_val, train=True)
    trainloader, valloader = Data.get_loader()
    TestData = MIL_dataloader(data_path=test_path, train=False)
    testloader = TestData.get_loader()

    early_stopping = EarlyStopping(model_path=model_save_path,
                                   patience=20, verbose=True) # 15
    scheduler = StepLR(optimizer=optimizer, step_size=30, verbose=True)

    val_losses = []
    best_model_path = None
    best_f1 = 0
    for epoch in range(num_epochs):
        train_epoch(epoch, model, optimizer, trainloader, loss_f)
        valid_loss, val_f1 = prediction(model, valloader, loss_f)
        scheduler.step()
        val_losses.append(valid_loss)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_path = model_save_path.replace('.pth', '_epoch_{}.pth'.format(epoch))
        
        if epoch % 10 == 0 or epoch == num_epochs - 1: 
            torch.save(model.state_dict(), best_model_path)
            print("Best model in the {} epoches is: {}".format(epoch, best_model_path))

    
    model_test = HACL_Net().cuda()
    model_test.load_state_dict(torch.load(best_model_path))

    loss, f1 = prediction(model_test, testloader, loss_f, testing=True)

    return f1


if __name__ == '__main__':

    args = parser.parse_args()

    img_label_path = args.img_label_path
    num_epochs = args.nepochs
    feat_path = args.feat_path
    if args.debug == True: num_epochs = 1
    print("Debug: ", args.debug)
    print("Epochs: ", num_epochs)
    lr = args.lr

    all_paths = pd.read_csv(img_label_path)

    label = all_paths['label'].tolist()
    for i in range(len(label)):
        if int(label[i]) == 3 or int(label[i]) == 2: 
            label[i] = 1
    pid = all_paths['pid'].tolist()

    uniq_pid = np.unique(pid)
    uniq_label = []

    for each_pid in uniq_pid:
        temp = pid.index(each_pid)
        uniq_label.append(label[temp])

    testaccur = []
    index_num = 1

    pid_ind = range(len(uniq_label))

    kf = StratifiedKFold(n_splits=5, random_state=666, shuffle=True)
    fold = 0
    for train_index, test_index in kf.split(pid_ind, uniq_label):
        print('-'*99)
        print("Now training fold: {}".format(fold))

        test_pid = [uniq_pid[i] for i in test_index]

        train_val_npz = [str(uniq_pid[i])+'.npz' for i in train_index]
        test_npz = [str(uniq_pid[i])+'.npz' for i in test_index]

        train_val_patients_pca = [os.path.join(feat_path , each_path) for each_path in train_val_npz]
        test_patients_pca = [os.path.join(feat_path, each_path) for each_path in test_npz]

        uniq_label_train_val = np.array([uniq_label[i] for i in train_index])

        model_save_path = '/media/data0/placenta/saved_models/final_model/NLST_model_{}_fold_{}_softmax.pth'.format(args.l1_reg, fold)
        test_accur = train(train_val_patients_pca, test_patients_pca, model_save_path, uniq_label_train_val, num_epochs=num_epochs, lr=lr)

        testaccur.append(test_accur)

        fold += 1


    print(testaccur)
    print(np.mean(testaccur))
