import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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



class LSTMDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.labels[self.labels<0] = 0  # label {1,-1} to label {1,0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    

    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)
        return out[:,0]
    
    
class NavalModel1(torch.nn.Module):
    def __init__(self, nf, nc, length, weight_bi, weight_l1):
        super().__init__()
        self.nf = nf
        self.nc = nc
        self.length = length
        self.t1 = torch.nn.Parameter(torch.randint(0,length//2,(nf,),dtype=torch.float32,requires_grad=True))
        self.t2 = torch.nn.Parameter(torch.randint(length//2,length,(nf,),dtype=torch.float32,requires_grad=True))
        self.a = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.p = torch.nn.Parameter(torch.rand((nc,nf), dtype=torch.float32, requires_grad=True))
        self.tvar_temporal = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.tvar_logical = torch.nn.Parameter(torch.rand(nc, dtype=torch.float32, requires_grad=True))
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
            self.tvar_temporal.clamp_(0,1)
            self.tvar_logical.clamp_(0,1)
            self.t1.clamp_(0,self.length-1)
            self.t2.clamp_(0,self.length-1)
            self.t1[self.t1>=self.t2-1] = self.t2[self.t1>=self.t2-1]-1
            self.eps[self.eps<0] = 1e-5
        
        batch_size = x.shape[0]
        r1 = torch.empty((batch_size,self.nf))
        for k, (predi, Ti) in enumerate(zip(self.pred, self.temporal)):
            rp = predi.forward(x) # predicate
            r1[:,k] = Ti.forward(rp,padding=False) # temporal
        r2 = torch.empty((batch_size,self.nc))
        for k, li in enumerate(self.logical):
            r2[:,k] = li.forward(r1,self.p[k,:],keepdim=False)
        r = r2[:,0]
        bi_reg = self.reg_bi.get_reg([self.tvar_temporal, self.tvar_logical, self.p]) # bi-modal regularizer
        l1_reg = self.reg_l1.get_reg([self.p]) # l1 regularizer
        reg = bi_reg + l1_reg
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
            formula_T = [] # temporal operator
            formula_time = [] # time interval
            formula_str = []
            formula_symbol = [] # >, <
            formula_pred = []
            str_list = ['x','y']
            w = torch.tensor(range(self.length), requires_grad=False)
            for k, (predi, T) in enumerate(zip(self.pred, self.temporal)):
                if T.type_var>0.5:
                    formula_T.append('F')
                else:
                    formula_T.append('G')
                formula_time.append([int(T.t1),int(T.t2)])
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
            for index, i in enumerate(logical_index):
                if index > 0 and len(logical_index)>1:
                    if self.logical[0].type_var>0.5:
                        print(' \u2228 ')
                    else:
                        print(' \u2227 ')
                print(formula_T[i]+"["+str(formula_time[i][0])+","+str(formula_time[i][1])+"]"+formula_str[i]+formula_symbol[i]+'{:.2f}'.format(formula_pred[i]))



class NavalModel2(torch.nn.Module):
    def __init__(self, nf, nc1, nc2, length, weight_bi, weight_l1):
        super().__init__()
        self.nf = nf
        self.nc1 = nc1
        self.nc2 = nc2
        self.length = length
        self.a = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.t1 = torch.nn.Parameter(torch.randint(0,length//2,(nf,),dtype=torch.float32,requires_grad=True))
        self.t2 = torch.nn.Parameter(torch.randint(length//2,length,(nf,),dtype=torch.float32,requires_grad=True))
        self.p1 = torch.nn.Parameter(torch.rand((nc1,nf), dtype=torch.float32, requires_grad=True))
        self.p2 = torch.nn.Parameter(torch.rand((nc2,nc1), dtype=torch.float32, requires_grad=True))
        self.tvar_temporal = torch.nn.Parameter(torch.rand(nf, dtype=torch.float32, requires_grad=True))
        self.tvar_logical1 = torch.nn.Parameter(torch.rand(nc1, dtype=torch.float32, requires_grad=True))
        self.tvar_logical2 = torch.nn.Parameter(torch.rand(nc2, dtype=torch.float32, requires_grad=True))
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
        for i in range(nc1):
            self.logical1.append(LogicalOperator(logical_type,dim=1,avm=True,beta=False,type_var=self.tvar_logical1[i]))
        self.logical2 = []
        for i in range(nc2):
            self.logical2.append(LogicalOperator(logical_type,dim=1,avm=True,beta=False,type_var=self.tvar_logical2[i]))
        self.reg_bi = Bimodal_reg(weight_bi)
        self.reg_l1 = L1_reg(weight_l1)
        self.eps = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        with torch.no_grad():
            self.p1.clamp_(0, 1)
            for i in range(self.p1.shape[0]):
                if torch.all(self.p1[i,:]==0) and self.p2[0,i]>0.5:
                    p_new = torch.rand(self.p1[i,:].shape)
                    self.p1[i,:].data.copy_(p_new)
            self.p2.clamp_(0, 1)
            if torch.all(self.p2==0):
                p_new = torch.rand(self.p2.shape)
                self.p2.data.copy_(p_new)
            self.tvar_temporal.clamp_(0,1)
            self.tvar_logical1.clamp_(0,1)
            self.tvar_logical2.clamp_(0,1)
            self.t1.clamp_(0,self.length-1)
            self.t2.clamp_(0,self.length-1)
            self.t1[self.t1>=self.t2-1] = self.t2[self.t1>=self.t2-1]-1
            self.eps[self.eps<0] = 1e-5
        
        batch_size = x.shape[0]
        r1 = torch.empty((batch_size,self.nf))
        for k, (predi, Ti) in enumerate(zip(self.pred, self.temporal)):
            rp = predi.forward(x) # predicate
            r1[:,k] = Ti.forward(rp,padding=False) # temporal
        r2 = torch.empty((batch_size,self.nc1))
        for k, li in enumerate(self.logical1):
            r2[:,k] = li.forward(r1,self.p1[k,:],keepdim=False)
        r3 = torch.empty((batch_size,self.nc2))
        for k, li in enumerate(self.logical2):
            r3[:,k] = li.forward(r2,self.p2[k,:],keepdim=False)
        r = r3[:,0]
        bi_reg = self.reg_bi.get_reg([self.tvar_temporal, self.tvar_logical1, self.tvar_logical2, self.p1, self.p2]) # bi-modal regularizer
        l1_reg = self.reg_l1.get_reg([self.p1,self.p2]) # l1 regularizer
        reg = bi_reg + l1_reg
        return r, reg
    
    def accuracy_formula(self, x, y):
        '''
        Classification accuracy of the learned STL formula
        '''
        with torch.no_grad():
            batch_size = x.shape[0]
            r1 = torch.empty((batch_size,self.nf))
            for k, (predi, Ti) in enumerate(zip(self.pred, self.temporal)):
                rp = predi.forward(x) # predicate
                r1[:,k] = Ti.formula_forward(rp,padding=False) # temporal
            r2 = torch.empty((batch_size,self.nc1))
            p1b = STEstimator.apply(self.p1)
            for k, li in enumerate(self.logical1):
                r2[:,k] = li.formula_forward(r1,p1b[k,:],keepdim=False)
            r3 = torch.empty((batch_size,self.nc2))
            p2b = STEstimator.apply(self.p2)
            for k, li in enumerate(self.logical2):
                r3[:,k] = li.formula_forward(r2,p2b[k,:],keepdim=False)
            r = r3[:,0]
            y_pred = Clip.apply(r)
            acc = torch.sum((y_pred==y))/batch_size
        return acc
    
    def translate_formula(self):
        '''
        Translate the whole network to an STL formula
        '''
        with torch.no_grad():
            str_list = ['x','y']
            w = torch.tensor(range(self.length), requires_grad=False)
            formula = []
            for k, (predi, T) in enumerate(zip(self.pred, self.temporal)):
                formula_i = []
                if T.type_var>0.5:
                    formula_i = 'F'+'['+str(int(T.t1))+','+str(int(T.t2))+']'
                else:
                    formula_i = 'G'+'['+str(int(T.t1))+','+str(int(T.t2))+']'
                d = predi.dim
                formula_i = formula_i + str_list[d]
                if predi.a>0:
                    formula_i = formula_i + '>' + '{:.2f}'.format(predi.b.item()/predi.a.item())
                else:
                    formula_i = formula_i + '<' + '{:.2f}'.format(predi.b.item()/predi.a.item())
                formula.append(formula_i)
            p1b = STEstimator.apply(self.p1)
            p2b = STEstimator.apply(self.p2)
            logical2_index = torch.where(torch.squeeze(p2b[0,:])==1)[0]
            for index2, j in enumerate(logical2_index):
                if index2 > 0 and len(logical2_index)>1:
                    if self.logical2[0].type_var>0.5:
                        print(' \u2228 ')
                    else:
                        print(' \u2227 ')
                logical1_index = torch.where(torch.squeeze(p1b[j,:])==1)[0]
                formula_i = ''
                for index1, i in enumerate(logical1_index):
                    if index1 > 0 and len(logical1_index)>1:
                        if self.logical1[j].type_var>0.5:
                            formula_i = formula_i + ' \u2228 '
                        else:
                            formula_i = formula_i + ' \u2227 '
                    formula_i = formula_i + formula[i]
                print(formula_i)



class NavalModel3(torch.nn.Module):
    def __init__(self, nf, nc1, nc2, length, weight_bi, weight_l1):
        super().__init__()
        self.nf = nf
        self.nc1 = nc1
        self.nc2 = nc2
        self.length = length
        self.a = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.t1 = torch.nn.Parameter(torch.randint(0,length//2,(nc1,),dtype=torch.float64,requires_grad=True))
        self.t2 = torch.nn.Parameter(torch.randint(length//2,length,(nc1,),dtype=torch.float64,requires_grad=True))
        self.p1 = torch.nn.Parameter(torch.ones((nc1,nf), dtype=torch.float64, requires_grad=True))
        self.p2 = torch.nn.Parameter(torch.ones((nc2,nc1), dtype=torch.float64, requires_grad=True))
        self.tvar_temporal = torch.nn.Parameter(torch.rand(nf, dtype=torch.float64, requires_grad=True))
        self.tvar_logical1 = torch.nn.Parameter(torch.rand(nc1, dtype=torch.float64, requires_grad=True))
        self.tvar_logical2 = torch.nn.Parameter(torch.rand(nc2, dtype=torch.float64, requires_grad=True))
        variable_based = True # whether to learn the type of operators
        if variable_based:
            temporal_type = ['temporal' for i in range(nf)]
            logical_type = 'logical'
        self.tau = torch.tensor(1, requires_grad=False) # slope of time function
        self.pred = []
        for i in range(nf):
            self.pred.append(Predicate(self.a[i],self.b[i],dim=i%2))
        self.logical1 = []
        for i in range(nc1):
            self.logical1.append(LogicalOperator(logical_type,dim=2,avm=True,beta=False,type_var=self.tvar_logical1[i]))
        self.temporal = []
        for i in range(nc1):
            self.temporal.append(TemporalOperator(temporal_type[i],self.tau,self.t1[i],self.t2[i],beta=2.5,h=1,type_var=self.tvar_temporal[i]))
        self.logical2 = []
        for i in range(nc2):
            self.logical2.append(LogicalOperator(logical_type,dim=1,avm=True,beta=False,type_var=self.tvar_logical2[i]))
        self.reg_bi = Bimodal_reg(weight_bi)
        self.reg_l1 = L1_reg(weight_l1)
        self.eps = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64, requires_grad=True))

    def forward(self, x):
        with torch.no_grad():
            self.p1.clamp_(0, 1)
            for i in range(self.p1.shape[0]):
                if torch.all(self.p1[i,:]==0) and self.p2[0,i]>0.5:
                    p_new = torch.rand(self.p1[i,:].shape)
                    self.p1[i,:].data.copy_(p_new)
            self.p2.clamp_(0, 1)
            if torch.all(self.p2==0):
                p_new = torch.rand(self.p2.shape)
                self.p2.data.copy_(p_new)
            self.tvar_temporal.clamp_(0,1)
            self.tvar_logical1.clamp_(0,1)
            self.tvar_logical2.clamp_(0,1)
            self.t1.clamp_(0,self.length-1)
            self.t2.clamp_(0,self.length-1)
            self.t1[self.t1>=self.t2-1] = self.t2[self.t1>=self.t2-1]-1
            self.eps[self.eps<0] = 1e-5
        
        batch_size = x.shape[0]
        rp = torch.empty((batch_size,self.length,self.nf))
        for k, predi in enumerate(self.pred):
            rp[:,:,k] = predi.forward(x) # predicate
        r1 = torch.empty((batch_size,self.nc1))
        for k, (li, Ti) in enumerate(zip(self.logical1, self.temporal)):
            rl = li.forward(rp,self.p1[k,:],keepdim=False) # logical
            r1[:,k] = Ti.forward(rl,padding=False) # temporal
        r2 = torch.empty((batch_size,self.nc2))
        for k, li in enumerate(self.logical2):
            r2[:,k] = li.forward(r1,self.p2[k,:],keepdim=False)
        r = r2[:,0]
        bi_reg = self.reg_bi.get_reg([self.tvar_temporal, self.tvar_logical1, self.tvar_logical2, self.p1, self.p2]) # bi-modal regularizer
        l1_reg = self.reg_l1.get_reg([self.p1,self.p2]) # l1 regularizer
        reg = bi_reg + l1_reg
        return r, reg
    
    def accuracy_formula(self, x, y):
        '''
        Classification accuracy of the learned STL formula
        '''
        with torch.no_grad():
            batch_size = x.shape[0]
            rp = torch.empty((batch_size,self.length,self.nf))
            for k, predi in enumerate(self.pred):
                rp[:,:,k] = predi.forward(x) # predicate
            r1 = torch.empty((batch_size,self.nc1))
            p1b = STEstimator.apply(self.p1)
            for k, (li, Ti) in enumerate(zip(self.logical1, self.temporal)):
                rl = li.formula_forward(rp,p1b[k,:],keepdim=False)
                r1[:,k] = Ti.formula_forward(rl,padding=False) # temporal
            r2 = torch.empty((batch_size,self.nc2))
            p2b = STEstimator.apply(self.p2)
            for k, li in enumerate(self.logical2):
                r2[:,k] = li.formula_forward(r1,p2b[k,:],keepdim=False)
            r = r2[:,0]
            y_pred = Clip.apply(r)
            acc = torch.sum((y_pred==y))/batch_size
        return acc
    
    def translate_formula(self):
        '''
        Translate the whole network to an STL formula
        '''
        with torch.no_grad():
            formula_str = []
            formula_symbol = [] # >, <
            formula_pred = []
            str_list = ['x','y']
            w = torch.tensor(range(self.length), requires_grad=False)
            for k, predi in enumerate(self.pred):
                d = predi.dim
                formula_str.append(str_list[d])
                if predi.a>0:
                    formula_symbol.append('>')
                    formula_pred.append(predi.b.item()/predi.a.item())
                else:
                    formula_symbol.append('<')
                    formula_pred.append(predi.b.item()/predi.a.item())
            p1b = STEstimator.apply(self.p1)
            formula = []
            for k, (li, T) in enumerate(zip(self.logical1, self.temporal)):
                formula_i = ''
                logical1_index = torch.where(torch.squeeze(p1b[k,:])==1)[0]
                if len(logical1_index)==0:
                    formula.append(formula_i)
                    continue
                if T.type_var>0.5:
                    formula_i = formula_i + 'F'
                else:
                    formula_i = formula_i + 'G'
                formula_i = formula_i + '['+str(int(T.t1))+','+str(int(T.t2))+']'
                
                for index1, i in enumerate(logical1_index):
                    if index1 > 0 and len(logical1_index)>1:
                        if li.type_var>0.5:
                            formula_i = formula_i + ' \u2228 '
                        else:
                            formula_i = formula_i + ' \u2227 '
                    formula_i = formula_i + formula_str[i]+formula_symbol[i]+'{:.2f}'.format(formula_pred[i])
                formula.append(formula_i)
            p2b = STEstimator.apply(self.p2)
            logical2_index = torch.where(torch.squeeze(p2b[0,:])==1)[0]
            formula_prev = True
            for index2, j in enumerate(logical2_index):
                if formula[j] == '':
                    formula_prev = False
                    continue
                else:
                    if index2 > 0 and len(logical2_index)>1 and formula_prev==True:
                        formula_prev = True
                        if self.logical2[0].type_var>0.5:
                            print(' \u2228 ')
                        else:
                            print(' \u2227 ')
                print(formula[j])