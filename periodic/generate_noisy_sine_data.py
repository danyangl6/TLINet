import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import pickle
from random import shuffle

'''
Generate noisy sine data as training data, and use sine data without noise as validation data.
'''

# initialization
num_paths = 1000
length = 60

# paths with positive label +1
positive_paths = []
positive_paths_noisy = []
T = 20
x = np.linspace(0, 2*length, 2*length+1)
sine = np.sin(2*np.pi/T * x)
for i in range(num_paths):
    start = np.random.randint(low=0,high=length)
    A = np.random.uniform(low=1.0,high=5.0)
    pos_path = sine[start:start+length]*A
    noise = np.random.normal(0,A/10,(length,1))
    target_snr_db = 15
    x_watts = pos_path ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    std = np.sqrt(noise_avg_watts)
    noise = np.random.normal(0,std,(length,1))
    pos_path_noisy = np.expand_dims(pos_path, axis=1) + noise
    positive_paths.append(np.expand_dims(pos_path, axis=1))
    positive_paths_noisy.append(pos_path_noisy)

# paths with negative label -1
negative_paths = []
negative_paths_noisy = []
T = 40
x = np.linspace(0, 2*length, 2*length+1)
sine = np.sin(2*np.pi/T * x)
for i in range(num_paths):
    start = np.random.randint(low=0,high=length)
    A = np.random.uniform(low=1.0,high=5.0)
    pos_path = sine[start:start+length]*A
    noise = np.random.normal(0,0.1,(length,1))
    target_snr_db = 15
    x_watts = pos_path ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    std = np.sqrt(noise_avg_watts)
    noise = np.random.normal(0,std,(length,1))
    pos_path_noisy = np.expand_dims(pos_path, axis=1) + noise
    negative_paths.append(np.expand_dims(pos_path, axis=1))
    negative_paths_noisy.append(pos_path_noisy)

positive_paths = np.array(positive_paths)
negative_paths = np.array(negative_paths)
x = np.append(positive_paths,negative_paths,axis=0)
y = np.append(np.ones(num_paths),-np.ones(num_paths))
ind_list = [i for i in range(num_paths*2)]
shuffle(ind_list)
data  = torch.tensor(x[ind_list, :,:])
label = torch.tensor(y[ind_list])

positive_paths_noisy = np.array(positive_paths_noisy)
negative_paths_noisy = np.array(negative_paths_noisy)
x = np.append(positive_paths_noisy,negative_paths_noisy,axis=0)
y = np.append(np.ones(num_paths),-np.ones(num_paths))
noisy_data  = torch.tensor(x[ind_list, :,:])
noisy_label = torch.tensor(y[ind_list])

print('dataset shape: '+ str(noisy_data.shape))

f = open('noisy_sine_dataset.pkl', 'wb')
pickle.dump([noisy_data, noisy_label, data, label], f)
f.close()

plt.figure(figsize=(8, 5))
for i in range(10):
    path = data[i,:,:]
    noisy_path = noisy_data[i,:,:]
    label = noisy_label[i]
    if label == 1:
        p1 = plt.plot(path[:,0], color='red', label='1')
        p3 = plt.plot(noisy_path[:,0], '--', color='red',label='1')
    else:
        p2 = plt.plot(path[:,0], color='blue',label='-1')
        p4 = plt.plot(noisy_path[:,0], '--', color='blue',label='-1')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("noisy sine dataset", fontsize=16)
figname = "noisy_sine.png"
plt.savefig(figname)
plt.show()

