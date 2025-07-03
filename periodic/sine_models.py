import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import random
import time
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
    def __init__(self, nf, nc, length, weight_bi, weight_l1):
        super().__init__()
        self.nf = nf
        self.nc = nc
        self.length = length
        self.t11 = torch.nn.Parameter(torch.randint(0,length//2,(nf,),dtype=torch.float32,requires_grad=True))
        self.t12 = torch.nn.Parameter(torch.randint(length//2,length,(nf,),dtype=torch.float32,requires_grad=True))
        self.t21 = torch.nn.Parameter(torch.randint(0,length//2,(nf,),dtype=torch.float32,requires_grad=True))
        self.t22 = torch.nn.Parameter(torch.randint(length//2,length,(nf,),dtype=torch.float32,requires_grad=True))
        self.a = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.p = torch.nn.Parameter(torch.rand((nc,nf), dtype=torch.float32, requires_grad=True))
        self.tvar_temporal1 = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.tvar_temporal2 = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.tvar_logical = torch.nn.Parameter(torch.rand(nc, dtype=torch.float32, requires_grad=True))
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
            self.temporal1.append(TemporalOperator(temporal_type[i],self.tau,self.t11[0],self.t12[i],beta=2.5,h=1,type_var=self.tvar_temporal1[i]))
        self.temporal2 = []
        for i in range(nf):
            self.temporal2.append(TemporalOperator(temporal_type[i],self.tau,self.t21[i],self.t22[i],beta=2.5,h=1,type_var=self.tvar_temporal2[i]))
        self.logical = []
        for i in range(nc):
            self.logical.append(LogicalOperator(logical_type,dim=1,avm=True,beta=False,type_var=self.tvar_logical[i]))
        self.reg_bi = Bimodal_reg(weight_bi)
        self.reg_l1 = L1_reg(weight_l1)
        self.eps = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        with torch.no_grad():
            self.p.clamp_(0, 1)
            for i in range(self.p.shape[0]):
                if torch.all(self.p[i,:]==0):
                    p_new = torch.rand(self.p[i,:].shape)
                    self.p[i,:].data.copy_(p_new)
            self.tvar_temporal1.clamp_(0,1)
            self.tvar_temporal2.clamp_(0,1)
            self.tvar_logical.clamp_(0,1)
            self.t11.clamp_(0,self.length-1)
            self.t12.clamp_(0,self.length-1)
            self.t21.clamp_(0,self.length-1)
            self.t22.clamp_(0,self.length-1)
            self.t11[self.t11>=self.t12-1] = self.t12[self.t11>=self.t12-1]-1
            self.t21[self.t21>=self.t22-1] = self.t22[self.t21>=self.t22-1]-1
            self.eps[self.eps<0] = 1e-5
        
        batch_size = x.shape[0]
        r1 = torch.empty((batch_size,self.nf))
        for k, (predi, T1, T2) in enumerate(zip(self.pred, self.temporal1, self.temporal2)):
            r1p = predi.forward(x)
            r1i = T1.forward(r1p,padding=True)
            r1[:,k] = T2.forward(r1i,padding=False)
        r2 = torch.empty((batch_size,self.nc))
        for k, li in enumerate(self.logical):
            r2[:,k] = li.forward(r1,self.p[k,:],keepdim=False)
        r = r2[:,0]
        bi_reg = self.reg_bi.get_reg([self.tvar_temporal1, self.tvar_temporal2, self.tvar_logical, self.p]) # bi-modal regularizer
        l2_reg = self.reg_l1.get_reg([self.p]) # l1 regularizer
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
            r2 = torch.empty((batch_size,self.nc))
            for k, li in enumerate(self.logical):
                r2[:,k] = li.formula_forward(r1,pb[k,:],keepdim=False)
            r = r2[:,0]
            y_pred = Clip.apply(r)
            acc = torch.sum((y_pred==y))/batch_size
        return acc
    
    def translate_formula(self):
        '''
        Translate the whole network to an STL formula
        '''
        with torch.no_grad():
            str_list = ['x','y']
            formula = []
            for k, (predi, T1, T2) in enumerate(zip(self.pred, self.temporal1, self.temporal2)):
                formula_i = ''
                if T2.type_var>0.5:
                    formula_i = formula_i + 'F' + '['+str(int(T2.t1))+','+str(int(T2.t2))+']'
                else:
                    formula_i = formula_i + 'G' + '['+str(int(T2.t1))+','+str(int(T2.t2))+']'
                if T1.type_var>0.5:
                    formula_i = formula_i + 'F' + '['+str(int(T1.t1))+','+str(int(T1.t2))+']'
                else:
                    formula_i = formula_i + 'G' + '['+str(int(T1.t1))+','+str(int(T1.t2))+']'
                d = predi.dim
                formula_i = formula_i + str_list[d]
                if predi.a>0:
                    formula_i = formula_i + '>' + '{:.2f}'.format(predi.b.item()/predi.a.item())
                else:
                    formula_i = formula_i + '<' + '{:.2f}'.format(predi.b.item()/predi.a.item())
                formula.append(formula_i)
            pb = STEstimator.apply(self.p)
            logical_index = torch.where(torch.squeeze(pb)==1)[0]
            for index, j in enumerate(logical_index):
                if index > 0 and len(logical_index)>1:
                    if self.logical[0].type_var>0:
                        print(' \u2228 ')
                    else:
                        print(' \u2227 ')
                print(formula[j])