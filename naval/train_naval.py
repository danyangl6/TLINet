import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import random
from random import shuffle
import sys
sys.path.append("..")
from neurons import *


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class NavalModel(torch.nn.Module):
    def __init__(self, nf, nc, length, weight_bi, weight_l2):
        super().__init__()
        self.nf = nf
        self.nc = nc
        self.length = length
        self.t1 = torch.nn.Parameter(torch.randint(0,length//2,(nf,),dtype=torch.float64,requires_grad=True))
        self.t2 = torch.nn.Parameter(torch.randint(length//2,length,(nf,),dtype=torch.float64,requires_grad=True))
        self.a = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.p1 = torch.nn.Parameter(torch.rand((nc,nf), dtype=torch.float64, requires_grad=True))
        self.p2 = torch.nn.Parameter(torch.rand((1,nc), dtype=torch.float64, requires_grad=True))
        self.tvar_temporal = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.tvar_logical1 = torch.nn.Parameter(torch.rand(nc, dtype=torch.float64, requires_grad=True))
        self.tvar_logical2 = torch.nn.Parameter(torch.rand(1, dtype=torch.float64, requires_grad=True))
        variable_based = True # whether to learn the type of operators
        if variable_based:
            temporal_type = ['temporal' for i in range(nf)]
            logical_type = 'logical'
        self.tau = torch.tensor(1, requires_grad=False) # slope of time function
        self.pred = []
        for i in range(nf):
            self.pred.append(Predicate(self.a[i],self.b[i],dim=i%2))
        self.temporal = []
        for i in range(nf):
            self.temporal.append(TemporalOperator(temporal_type[i],self.tau,self.t1[i],self.t2[i],beta=2.5,h=1,type_var=self.tvar_temporal[i]))
        self.logical1 = []
        for i in range(nc):
            self.logical1.append(LogicalOperator(logical_type,dim=1,avm=True,beta=False,type_var=self.tvar_logical1[i]))
        self.logical2 = LogicalOperator(logical_type,dim=1,avm=True,beta=False,type_var=self.tvar_logical2)
        self.reg_bi = Bimodal_reg(weight_bi)
        self.reg_l2 = L2_reg(weight_l2)
        self.eps = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64, requires_grad=True))

    def forward(self, x):
        with torch.no_grad():
            self.p1.clamp_(0, 1)
            for i in range(self.p1.shape[0]):
                if torch.all(self.p1[i,:]==0):
                    p_new = torch.rand(self.p1[i,:].shape)
                    self.p1[i,:].data.copy_(p_new)
            self.p2.clamp_(0, 1)
            if torch.all(self.p2==0):
                p_new = torch.rand(self.p2.shape)
                self.p2.data.copy_(p_new)
            self.tvar_temporal.clamp_(0,1)
            self.tvar_logical1.clamp_(0,1)
            self.tvar_logical2.clamp_(0,1)
            self.t1.clamp_(0,length-1)
            self.t2.clamp_(0,length-1)
            self.t1[self.t1>=self.t2-1] = self.t2[self.t1>=self.t2-1]-1
            self.eps[self.eps<0] = 1e-5
        
        batch_size = x.shape[0]
        r1 = torch.empty((batch_size,self.nf))
        for k, (predi, Ti) in enumerate(zip(self.pred, self.temporal)):
            rp = predi.forward(x) # predicate
            r1[:,k] = Ti.forward(rp,padding=False) # temporal
        r2 = torch.empty((batch_size,nc))
        for k, li in enumerate(self.logical1):
            r2[:,k] = li.forward(r1,self.p1[k,:],keepdim=False)
        r = self.logical2.forward(r2,self.p2[0,:],keepdim=False)
        bi_reg = self.reg_bi.get_reg([self.tvar_temporal, self.tvar_logical1, self.tvar_logical2, self.p1, self.p2]) # bi-modal regularizer
        l2_reg = self.reg_l2.get_reg([self.p1,self.p2]) # l2 regularizer
        reg = bi_reg + l2_reg
        return r, reg
    
    def accuracy_formula(self, x, y):
        '''
        Classification accuracy of the learned STL formula
        '''
        with torch.no_grad():
            batch_size = x.shape[0]
            r1 = torch.empty((batch_size,self.nf))
            for k, (predi, Ti) in enumerate(zip(self.pred, self.temporal)):
                rp = predi.forward(x)
                r1[:,k] = Ti.formula_forward(rp,padding=False)
            r2 = torch.empty((batch_size,nc))
            p1b = STEstimator.apply(self.p1)
            for k, li in enumerate(self.logical1):
                r2[:,k] = li.formula_forward(r1,p1b[k,:],keepdim=False)
            p2b = STEstimator.apply(self.p2)
            r = self.logical2.formula_forward(r2,p2b[0,:],keepdim=False)
            y_pred = Clip.apply(r)
            acc = torch.sum((y_pred==y))/batch_size
        return acc
    



# -----------Train inference network---------------------------------------
if __name__ == "__main__":
    random.seed(42)
    with open('naval_dataset.pkl', 'rb') as f:
        train_data, train_label, val_data, val_label = pickle.load(f)

    dataset = CustomDataset(train_data, train_label)
    batch_size_inf = 64
    dataloader = DataLoader(dataset, batch_size=batch_size_inf, shuffle=True)

    nsample = train_data.shape[0]
    val_nsample = val_data.shape[0]
    length = train_data.shape[1]
    dim = train_data.shape[2]
    nf = dim*3
    nc = 2
    weight_bi = [1e-1,1e-1,1e-1,1e-1,1e-1] # weight for bi-modal regularizer
    weight_l2 = [0,0] # weight for L2 regularizer
    weight_margin = 1e-3 # weight for margin

    # initialize
    epochs_inf = 1000
    relu = torch.nn.ReLU()
    stl = NavalModel(nf,nc,length,weight_bi,weight_l2)
    stl_optimizer = torch.optim.Adam(stl.parameters(), lr=1e-1)

    train_losses = {}
    stl.train()
    acc_best = 0
    for epoch in range(epochs_inf):
        epoch_losses = list()
        for batch_idx, (data_batch, labels_batch) in enumerate(dataloader):
            stl_optimizer.zero_grad()
            r, reg = stl(data_batch)
            loss_inf = torch.mean(relu(stl.eps-labels_batch*r)) - weight_margin*stl.eps + reg
            loss_inf.backward()
            stl_optimizer.step()

            epoch_losses.append(loss_inf.detach().item())

        train_losses[epoch] = torch.tensor(epoch_losses).mean()
        loss_log = list(train_losses.values())
        loss_log = [x.item() for x in loss_log]
        f = open('loss_log.pkl', 'wb')
        pickle.dump([loss_log], f)
        f.close()
        
        if (epoch+1) % 10 ==0:
            with torch.no_grad():
                r_train, _,  = stl(train_data)
                r_train = Clip.apply(r_train)
                acc_train = torch.sum((r_train==train_label))/nsample
                accf_train = stl.accuracy_formula(train_data, train_label)
                r_val, _,  = stl(val_data)
                r_val = Clip.apply(r_val)
                acc = torch.sum((r_val==val_label))/val_nsample
                accf = stl.accuracy_formula(val_data, val_label)
            print('epoch = {:3d}, loss_inf = {:.2f}, training network accuracy = {:.2f}, training formula accuracy = {:.2f}'.format(epoch+1,loss_inf,acc_train,accf_train))
            print('validation network accuracy = {:.2f}, validation formula accuracy = {:.2f}'.format(acc,accf))
            if accf_train>=acc_best:
                acc_best = accf_train
                torch.save(stl.state_dict(), f'naval_STL_model.pth')
                # stl.translate_formula() 
        stl_optimizer.zero_grad()
    print('Best training formula accuracy is {:.2f}'.format(acc_best))

    # plt.plot(range(1, epoch+2), loss_log, marker='o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Epoch')
    # plt.grid()
    # plt.show()
