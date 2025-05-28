import numpy as np
import pickle
import random
import torch
import matplotlib.pyplot as plt
from random import shuffle

if __name__ == "__main__":
    with open('naval_class.pkl', 'rb') as f:
        train_data, train_label, val_data, val_label = pickle.load(f)

    train_label[train_label!=1] = -1
    train_label[train_label!=-1] = 1
    val_label[val_label!=1] = -1
    val_label[val_label!=-1] = 1

    plt.figure(figsize=(8, 4))
    for j in range(val_data.shape[0]):
        if val_label[j] == 1:
            plt.plot(val_data[j,:,0],val_data[j,:,1], color = 'blue')
        else:
            plt.plot(val_data[j,:,0],val_data[j,:,1], color = 'red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Naval dataset")
    plt.grid()
    plt.show()

    train_data = torch.tensor(train_data,dtype=torch.float64)
    train_label = torch.tensor(train_label,dtype=torch.float64)
    val_data = torch.tensor(val_data,dtype=torch.float64)
    val_label = torch.tensor(val_label,dtype=torch.float64)

    with open('naval_dataset.pkl', "wb") as file:
        pickle.dump([train_data, train_label, val_data, val_label],file)