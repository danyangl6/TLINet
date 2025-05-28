import torch
import torch.nn as nn

class STEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        # g -> gs
        g_clip = torch.clamp(g, min = 0, max = 1)
        gs = g_clip.clone()
        gs[gs>=0.5] = 1
        gs[gs<0.5] = 0
        return gs
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input


class Clip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        gs = g.clone()
        gs[gs>=0] = 1
        gs[gs<0] = -1
        return gs
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input
    

class Bimodal_reg(object):
    def __init__(self, beta_list):
        '''
        beta_list is a list containing the weight for each probability regularizer
        '''
        if not isinstance(beta_list, list):
            raise TypeError('Weight for probability regularizer is not a list!')
        self.beta_list = beta_list
    def get_reg(self, p_list):
        if len(self.beta_list) != len(p_list):
            raise TypeError('Please specify weight for each layer!')
        r_all = []
        for _, (beta,p) in enumerate(zip(self.beta_list, p_list)):
            if len(p.shape) == 1:
                n_row = p.shape[0]
                r = 0
                for i in range(n_row):
                    pi = p[i]
                    r += pi*(1-pi)
                r_all.append(beta*r)
            else:
                n_row = p.shape[0]
                n_col = p.shape[1]
                r = 0
                for i in range(n_row):
                    for j in range(n_col):
                        pi = p[i,j]
                        r += pi*(1-pi)
                r_all.append(beta*r)
        return torch.stack(r_all).mean(dim=0)
    

class L2_reg(object):
    def __init__(self, beta_list):
        '''
        beta_list is a list containing the weight for each probability regularizer
        '''
        if not isinstance(beta_list, list):
            raise TypeError('Weight for probability regularizer is not a list!')
        self.beta_list = beta_list
    def get_reg(self, p_list):
        if len(self.beta_list) != len(p_list):
            raise TypeError('Please specify weight for each layer!')
        r_all = 0
        for _, (beta,p) in enumerate(zip(self.beta_list, p_list)):
            r_all += beta*torch.mean(torch.square(p))
        return r_all


class TimeFunction(object):
    def __init__(self):
        pass
    def forward(self, w, tau, t1, t2):
        relu = torch.nn.ReLU()
        f1 = (relu(w-t1+tau)-relu(w-t1))/tau
        f2 = (relu(-w+t2+tau)-relu(-w+t2))/tau
        w = torch.min(f1,f2)
        return w
  

class SparseSoftMax(object):
    def __init__(self, beta, h, dim):
        '''
        Specify the dimension to take the max
        beta,h are hyperparameters
        '''
        self.beta = beta
        self.h = h
        self.dim = dim
    def weight(self, x, w):
        '''
        generate weight for each element
        '''
        r_w = x*w
        mx = torch.abs(torch.max(r_w,dim=self.dim,keepdim=True)[0])
        if torch.any(mx==0):
            mx[mx==0] = 1
        r_norm = self.h*torch.div(r_w,mx) #rescale r
        r_exp = torch.exp(self.beta * r_norm)
        r_sum = torch.sum(r_exp,dim=self.dim,keepdim=True)
        s_norm = torch.div(r_exp,r_sum)
        return s_norm
    def forward(self, x, w, keepdim=False):
        '''
        x: input of size [batch_size, T, d] or [batch_size, d]
        w is a one-dimensional vector. 
        '''
        dim = self.dim
        if len(w.shape)>1:
            raise ValueError('Dimension of weight is invalid!')
        if dim != len(x.shape)-1:
            raise ValueError('Invalid operation!')
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum
        s_norm = self.weight(x, w_norm)
        sw = torch.mul(s_norm,w_norm)
        denominator = torch.sum(sw,dim=dim,keepdim=keepdim)
        numerator = torch.mul(sw, x)
        numerator = torch.sum(numerator,dim=dim,keepdim=keepdim)
        denominator_old = torch.clone(denominator)
        denominator[(denominator_old==0)] = 1
        rho = numerator/denominator
        return rho
        

class AveragedMax(object):
    def __init__(self, dim):
        '''
        Specify the dimension to take the max
        '''
        self.dim = dim
    def prob(self, p):
        '''
        the data (not the probability) to take the max is either [batch_size, T, d] (dim=2) or [batch_size, d] (dim=1)
        '''
        dim = self.dim
        if dim==2:
            prob = torch.empty(p.shape)
            for i in range(p.shape[dim]):
                pi = 1
                for j in range(i+1):
                    if j==i:
                        pi = pi*p[:,:,j]
                    else:
                        pi = pi*(1-p[:,:,j])
                prob[:,:,i] = pi
        if dim==1:
            prob = torch.empty(p.shape)
            for i in range(p.shape[dim]):
                pi = 1
                for j in range(i+1):
                    if j==i:
                        pi = pi*p[:,j]
                    else:
                        pi = pi*(1-p[:,j])
                prob[:,i] = pi
        return prob
    def forward(self, x, p, keepdim=False):
        '''
        x: input of size [batch_size, d] or [batch_size, T, d]
        p is a one-dimensional vector. 
        '''
        dim = self.dim
        if dim != len(x.shape)-1:
            raise ValueError('Invalid operation! Dimension mismatch!')
        xs, pindex = torch.sort(x,dim=dim,descending=True)
        psort = p[pindex]
        pw = self.prob(psort)
        expectation = torch.sum(xs*pw,dim=dim,keepdim=keepdim)
        return expectation
    

class Predicate(object):
    def __init__(self, a, b, dim=False):
        '''
        dim is specified if predicate is computed along that dimension, otherwise the predicate is a matrix computation. Default: False
        b is a 1-d scalar
        '''
        self.a = a
        self.b = b
        self.dim = dim
    def forward(self, x):
        '''
        x is of size [batch_size, T, d] where T is the length of signal.
        output is of size [batch_size, T].
        '''
        dim = self.dim
        if dim is False:
            predicate = torch.matmul(x,self.a) - self.b
        else:
            predicate = self.a*x[:,:,dim] - self.b
        return predicate
    

class LogicalOperator(object):
    def __init__(self, oper, dim, avm=True, beta=False, h=False, type_var=False):
        '''
        To specify the type of logical operator, oper = 'and' or 'or', type_var=False
        To learn the type of logical operator, oper = 'logical', the input variable 'type_var' is needed.
        If avm=True, then the averaged max is used, otherwise, the sparse softmax is used. Default: True
        To use sparse softmax, the input values 'beta' and 'h' are needed. Defalt: False
        '''
        self.operation = oper
        if self.operation == 'logical': # learn type of logical operator
            if type_var is False:
                raise ValueError('Missing variable for variable-based logical operator!')
            self.type_var = type_var
        elif self.operation == 'and':
            self.type_var = torch.tensor(0,dtype=torch.float64,requires_grad=False)
        elif self.operation == 'or':
            self.type_var = torch.tensor(1,dtype=torch.float64,requires_grad=False)
        else:
            raise ValueError("Logical operation type is invalid!")
        self.avm = avm
        if avm==True:
            self.max_function = AveragedMax(dim=dim)
        else:
            if beta is False:
                raise ValueError('Missing beta for sparse softmax function!')
            if h is False:
                raise ValueError('Missing h for sparse softmax function!')
            self.max_function = SparseSoftMax(beta=beta,h=h,dim=dim)
    def forward(self, x, w, keepdim=False):
        if self.avm == False:
            w = STEstimator.apply(w)
        xmin = (-1)*torch.clone(x)
        xmax = torch.clone(x)
        xrmin = self.max_function.forward(xmin,w,keepdim)
        xrmax = self.max_function.forward(xmax,w,keepdim)
        r = self.type_var*xrmax + (1-self.type_var)*(-1)*xrmin
        return r
    def formula_forward(self, x, w, keepdim=False):
        '''
        Compute the robustness of the translated formula
        '''
        with torch.no_grad():
            if self.avm == False:
                w = STEstimator.apply(w)
            if self.type_var>0.5: # if type_var>0, the Boolean operator is OR
                xmax = torch.clone(x)
                xrmax = self.max_function.forward(xmax,w,keepdim)
                r = xrmax
            else: # else, the Boolean operator is AND
                xmin = (-1)*torch.clone(x)
                xrmin = self.max_function.forward(xmin,w,keepdim)
                r = (-1)*xrmin
        return r
    

class TemporalOperator(object):
    def __init__(self, oper, tau, t1, t2, beta=False, h=False, type_var=False):
        '''
        To specify the type of logical operator, oper = 'F' or 'G', type_var=False
        To learn the type of logical operator, oper = 'temporal', the input variable 'type_var' is needed.
        TemporalOperator uses sparse softmax, the input values 'beta' and 'h' are needed.
        '''
        self.operation = oper
        if self.operation == 'temporal': # learn type of temporal operator
            if type_var is False:
                raise ValueError('Missing variable for learning temporal operator!')
            self.type_var = type_var
        elif self.operation == 'G': # always
            self.type_var = torch.tensor(0,dtype=torch.float64,requires_grad=False)
        elif self.operation == 'F': # eventually
            self.type_var = torch.tensor(1,dtype=torch.float64,requires_grad=False)
        else:
            raise ValueError("Temporal operation type is invalid!")
        if beta is False:
                raise ValueError('Missing beta for sparse softmax function!')
        if h is False:
            raise ValueError('Missing h for sparse softmax function!')
        else:
            self.max_function = SparseSoftMax(beta=beta,h=h,dim=1)
        self.tau = tau
        self.t1 = t1
        self.t2 = t2
        self.time_weight = TimeFunction()
    def padding(self, x):
        length = x.shape[1]
        rho_min = torch.min(x,dim=1)[0]
        rho_pad = torch.unsqueeze(rho_min,1).repeat((1,length-1))
        x_pad = torch.cat((x,rho_pad),dim=1)
        return x_pad
    def forward(self, x, padding=False):
        '''
        x is of size [batch_size, T] where T is the length of signal.
        If padding=False, output is of size [batch_size].
        If padding=True, output is of size [batch_size,T].
        '''
        if len(x.shape)!=2:
            raise ValueError('Input dimension is invalid!')
        xmin = (-1)*torch.clone(x)
        xmax = torch.clone(x)
        length = x.shape[-1]
        w = torch.tensor(range(length), requires_grad=False)
        wt = self.time_weight.forward(w,self.tau,self.t1,self.t2)
        if padding is False:
            xrmin = self.max_function.forward(xmin,wt,keepdim=False)
            xrmax = self.max_function.forward(xmax,wt,keepdim=False)
        else:
            x_padmin = self.padding(xmin)
            x_padmax = self.padding(xmax)
            xrmin = torch.empty(x.shape)
            xrmax = torch.empty(x.shape)
            for i in range(length):
                xi = torch.clone(x_padmin[:,i:(i+length)])
                ri = self.max_function.forward(xi,wt,keepdim=False)
                xrmin[:,i] = ri
            for i in range(length):
                xi = torch.clone(x_padmax[:,i:(i+length)])
                ri = self.max_function.forward(xi,wt,keepdim=False)
                xrmax[:,i] = ri
        r = self.type_var*xrmax + (1-self.type_var)*(-1)*xrmin
        return r
    def formula_forward(self, x, padding=False):
        '''
        Compute the robustness of the translated formula
        '''
        with torch.no_grad():
            length = x.shape[-1]
            w = torch.tensor(range(length), requires_grad=False)
            wt = self.time_weight.forward(w,self.tau,self.t1,self.t2)
            if self.type_var>0.5: # if type_var>0, the temporal operator is F
                xmax = torch.clone(x)
                if padding is False:
                    xrmax = self.max_function.forward(xmax,wt,keepdim=False)
                else:
                    x_padmax = self.padding(xmax)
                    xrmax = torch.empty(x.shape)
                    for i in range(length):
                        xi = torch.clone(x_padmax[:,i:(i+length)])
                        ri = self.max_function.forward(xi,wt,keepdim=False)
                        xrmax[:,i] = ri
                r = xrmax
            else: # else, the temporal operator is G
                xmin = (-1)*torch.clone(x)
                if padding is False:
                    xrmin = self.max_function.forward(xmin,wt,keepdim=False)
                else:
                    x_padmin = self.padding(xmin)
                    xrmin = torch.empty(x.shape)
                    for i in range(length):
                        xi = torch.clone(x_padmin[:,i:(i+length)])
                        ri = self.max_function.forward(xi,wt,keepdim=False)
                        xrmin[:,i] = ri
                r = (-1)*xrmin
        return r