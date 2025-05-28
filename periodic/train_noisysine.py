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


class SineModel(torch.nn.Module):
    def __init__(self, nf, nc, length, weight_bi, weight_l2):
        super().__init__()
        self.nf = nf
        self.nc = nc
        self.length = length
        self.t1 = torch.nn.Parameter(torch.randint(0,length//2,(2,nf),dtype=torch.float64,requires_grad=True))
        self.t2 = torch.nn.Parameter(torch.randint(length//2,length,(2,nf),dtype=torch.float64,requires_grad=True))
        self.a = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.p = torch.nn.Parameter(torch.rand((nc,nf), dtype=torch.float64, requires_grad=True))
        self.tvar_temporal = torch.nn.Parameter(torch.rand((2,nf), dtype=torch.float64, requires_grad=True))
        self.tvar_logical = torch.nn.Parameter(torch.rand(nc, dtype=torch.float64, requires_grad=True))
        variable_based = True # whether to learn the type of operators
        if variable_based:
            temporal_type = ['temporal' for i in range(nf)]
            logical_type = 'logical'
        self.tau = torch.tensor(1, requires_grad=False) # slope of time function
        self.pred = []
        for i in range(nf):
            self.pred.append(Predicate(self.a[i],self.b[i],dim=i%1))
        self.temporal1 = []
        for i in range(nf):
            self.temporal1.append(TemporalOperator(temporal_type[i],self.tau,self.t1[0,i],self.t2[0,i],beta=2.5,h=1,type_var=self.tvar_temporal[0,i]))
        self.temporal2 = []
        for i in range(nf):
            self.temporal2.append(TemporalOperator(temporal_type[i],self.tau,self.t1[1,i],self.t2[1,i],beta=2.5,h=1,type_var=self.tvar_temporal[1,i]))
        self.logical = LogicalOperator(logical_type,dim=1,avm=True,beta=False,type_var=self.tvar_logical)
        self.reg_bi = Bimodal_reg(weight_bi)
        self.reg_l2 = L2_reg(weight_l2)
        self.eps = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64, requires_grad=True))

    def forward(self, x):
        with torch.no_grad():
            self.p.clamp_(0, 1)
            if torch.all(self.p==0):
                p_new = torch.rand(self.p.shape)
                self.p.data.copy_(p_new)
            self.tvar_temporal.clamp_(0,1)
            self.tvar_logical.clamp_(0,1)
            self.t1.clamp_(0,length-1)
            self.t2.clamp_(0,length-1)
            self.t1[self.t1>=self.t2-1] = self.t2[self.t1>=self.t2-1]-1
            self.eps[self.eps<0] = 1e-5
        
        batch_size = x.shape[0]
        r1 = torch.empty((batch_size,self.nf))
        for k, (predi, T1, T2) in enumerate(zip(self.pred, self.temporal1, self.temporal2)):
            r1p = predi.forward(x)
            r1i = T1.forward(r1p,padding=True)
            r1[:,k] = T2.forward(r1i,padding=False)
        r = self.logical.forward(r1,self.p[0,:],keepdim=False)
        bi_reg = self.reg_bi.get_reg([self.tvar_temporal, self.tvar_logical, self.p]) # bi-modal regularizer
        l2_reg = self.reg_l2.get_reg([self.p]) # l2 regularizer
        reg = bi_reg + l2_reg
        return r, reg
    
    def accuracy_formula(self, x, y):
        '''
        Classification accuracy of the learned STL formula
        '''
        with torch.no_grad():
            batch_size = x.shape[0]
            r1 = torch.empty((batch_size,self.nf))
            for k, (predi, T1, T2) in enumerate(zip(self.pred, self.temporal1, self.temporal2)):
                r1p = predi.forward(x)
                r1i = T1.formula_forward(r1p,padding=True)
                r1[:,k] = T2.formula_forward(r1i,padding=False)
            pb = STEstimator.apply(self.p)
            r = self.logical.formula_forward(r1,pb[0,:],keepdim=False)
            y_pred = Clip.apply(r)
            acc = torch.sum((y_pred==y))/batch_size
        return acc
    
    def translate_formula(self):
        '''
        Translate the whole network to an STL formula
        '''
        with torch.no_grad():
            formula_T1 = [] # temporal operator
            formula_time1 = [] # time interval
            formula_T2 = [] # temporal operator
            formula_time2 = [] # time interval
            formula_str = []
            formula_symbol = [] # >, <
            formula_pred = []
            str_list = ['x','y']
            w = torch.tensor(range(self.length), requires_grad=False)
            for k, (predi, T1, T2) in enumerate(zip(self.pred, self.temporal1, self.temporal2)):
                if T1.type_var>0.5:
                    formula_T1.append('F')
                else:
                    formula_T1.append('G')
                formula_time1.append([int(self.t1[0,k]),int(self.t2[0,k])])
                if T2.type_var>0.5:
                    formula_T2.append('F')
                else:
                    formula_T2.append('G')
                formula_time2.append([int(self.t1[1,k]),int(self.t2[1,k])])
                d = predi.dim
                formula_str.append(str_list[d])
                if predi.a>0:
                    formula_symbol.append('>')
                    formula_pred.append(predi.b.item()/predi.a.item())
                else:
                    formula_symbol.append('<')
                    formula_pred.append(predi.b.item()/predi.a.item())
            pb = STEstimator.apply(self.p)
            logical_index = torch.where(torch.squeeze(pb)==1)[0]
            for indexc, j in enumerate(logical_index):
                if indexc > 0 and len(logical_index)>1:
                    if self.logical.type_var>0:
                        print(' \u2228 ')
                    else:
                        print(' \u2227 ')
                print(formula_T2[j]+"["+str(formula_time2[j][0])+","+str(formula_time2[j][1])+"]"
                    +formula_T1[j]+"["+str(formula_time1[j][0])+","+str(formula_time1[j][1])+"]"
                    +formula_str[j]+formula_symbol[j]+'{:.2f}'.format(formula_pred[j]))
            
    def extract_formula(self, train_data, train_label):
        '''
        Using training dataset to remove redundant subformulas
        '''
        acc = self.accuracy_formula(train_data, train_label)
        with torch.no_grad():
            p_buffer = torch.clone(self.p)
            for i in range(self.p.shape[1]):
                if self.p[0,i]<0.5:
                    continue
                else:
                    self.p[0,i] = 0
                    acci = self.accuracy_formula(val_data, val_label)
                    if acci == acc:
                        continue
                    else:
                        self.p[0,i] = 1
        return p_buffer
    



# -----------Train inference network---------------------------------------
if __name__ == "__main__":
    random.seed(42)
    with open('noisy_sine_dataset.pkl', 'rb') as f:
        train_data, train_label, val_data, val_label = pickle.load(f)

    dataset = CustomDataset(train_data, train_label)
    batch_size_inf = 64
    dataloader = DataLoader(dataset, batch_size=batch_size_inf, shuffle=True)

    nsample = train_data.shape[0]
    val_nsample = val_data.shape[0]
    length = train_data.shape[1]
    dim = train_data.shape[2]
    nf = dim*3
    nc = 1
    weight_bi = [1e-1,1e-1,1e-1] # weight for bi-modal regularizer
    weight_l2 = [0] # weight for L2 regularizer
    weight_margin = 1e-3 # weight for margin

    # initialize
    epochs_inf = 1000
    relu = torch.nn.ReLU()
    stl = SineModel(nf,nc,length,weight_bi,weight_l2)
    stl_optimizer = torch.optim.Adam(stl.parameters(), lr=1e-1)

    train_losses = {}
    stl.train()
    acc_best = 0
    for epoch in range(epochs_inf):
        epoch_losses = list()
        for batch_idx, (data_batch, labels_batch) in enumerate(dataloader):
            stl_optimizer.zero_grad()
            r, reg = stl(data_batch)
            loss_inf = torch.mean(relu(stl.eps-labels_batch*r)) - weight_margin*stl.eps #+ reg*epoch/10
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
                torch.save(stl.state_dict(), f'noisysine_STL_model.pth')
                stl.translate_formula() 
        stl_optimizer.zero_grad()
    print('Best training formula accuracy is {:.2f}'.format(acc_best))

    # plt.plot(range(1, epoch+2), loss_log, marker='o')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Epoch')
    # plt.grid()
    # plt.show()
