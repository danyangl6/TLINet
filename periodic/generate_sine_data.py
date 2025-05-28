import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import pickle
from random import shuffle

# initialization
num_paths = 1000
length = 60

# paths with positive label +1
positive_paths = []
T = 20
x = np.linspace(0, 2*length, 2*length+1)
sine = np.sin(2*np.pi/T * x)
for i in range(num_paths):
    start = np.random.randint(low=0,high=length)
    A = np.random.uniform(low=1.0,high=5.0)
    pos_path = sine[start:start+length]*A
    pos_path = np.expand_dims(pos_path, axis=1)
    positive_paths.append(pos_path)

# paths with negative label -1
negative_paths = []
T = 40
x = np.linspace(0, 2*length, 2*length+1)
sine = np.sin(2*np.pi/T * x)
for i in range(num_paths):
    start = np.random.randint(low=0,high=length)
    A = np.random.uniform(low=1.0,high=5.0)
    pos_path = sine[start:start+length]*A
    pos_path = np.expand_dims(pos_path, axis=1)
    negative_paths.append(pos_path)

positive_paths = np.array(positive_paths)
negative_paths = np.array(negative_paths)

# seperate training set and validation set
val_set = int(num_paths*0.2)
train_set = num_paths-val_set

x_train = np.append(positive_paths[0:train_set,:,:],negative_paths[0:train_set,:,:],axis=0)
x_val = np.append(positive_paths[num_paths-val_set:num_paths,:,:],negative_paths[num_paths-val_set:num_paths,:,:],axis=0)
y_train = np.append(np.ones(train_set),-np.ones(train_set))
y_val = np.append(np.ones(val_set),-np.ones(val_set))

train_set = train_set*2
val_set = val_set*2

ind_list = [i for i in range(train_set)]
shuffle(ind_list)
train_data  = torch.tensor(x_train[ind_list, :,:])
train_label = torch.tensor(y_train[ind_list])
print('training dataset shape: '+ str(train_data.shape))

ind_list = [i for i in range(val_set)]
shuffle(ind_list)
val_data  = torch.tensor(x_val[ind_list, :,:])
val_label = torch.tensor(y_val[ind_list])
print('validation dataset shape: ' + str(val_data.shape))

f = open('sine_dataset.pkl', 'wb')
pickle.dump([train_data, train_label, val_data, val_label], f)
f.close()

plt.figure(figsize=(8, 5))
for i in range(10):
    path = val_data[i,:,:]
    label = val_label[i]
    if label == 1:
        p1 = plt.plot(path[:,0], color='red',label='1')
    else:
        p2 = plt.plot(path[:,0], color='blue',label='-1')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)
plt.title("sine dataset", fontsize=16)
figname = "sine.png"
plt.savefig(figname)
plt.show()

