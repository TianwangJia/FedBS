import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.io as scio
import os
import sys

from scipy.linalg import fractional_matrix_power
import torchvision.transforms as transforms

sys.path.append('.')

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)


# Transform
class EA(object):
    def __init__(self):
        pass
    def __call__(self,x):
        new_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            cov = np.zeros((x.shape[1],x.shape[2],x.shape[2])) #(N,C,T)
            for j in range(x.shape[1]):
                cov[j] = np.cov(x[i,j])
            refEA = np.mean(cov,0)
            sqrtRefEA = fractional_matrix_power(refEA,-0.5)
            new_x[i] = np.matmul(sqrtRefEA,x[i])
        return new_x

class ArrayToTensor(object):
    def __init__(self) -> None:
        pass
    def __call__(self,x):
        return torch.from_numpy(x).type(torch.FloatTensor)

class ZScoreNorm(object):
    def __init__(self) -> None:
        pass
    def __call__(self,x):       
        new_x = np.zeros_like(x)
        for i in range(x.shape[0]):
            temp_x = x[i,0] #(C,T)
            for j in range(1,x.shape[1]): #N-1
                temp_x = np.concatenate((temp_x,x[i,j]),axis=1) #(C,NT)
            mean_c = np.mean(temp_x,axis=1,keepdims=True) 
            std_c = np.std(temp_x,axis=1,keepdims=True) 
            new_x[i] = (x[i]-mean_c)/std_c
        
        return new_x
            


class MIDataset(Dataset):
    def __init__(self, random_state, subject_id: list ,root='../data/BNCI2014001', mode='train', test_size=0.2,data_transform=None, label_transform=None) -> None:
        self.random_state = random_state
        self.subject_id = subject_id
        self.root = root
        self.mode = mode
        self.test_size = test_size
        self.data_transform = transforms.Compose(data_transform)
        self.label_transform =transforms.Compose(label_transform)

        X = []
        y = []
        for i in self.subject_id:
            data = scio.loadmat(self.root+'//'+str(i)+'.mat')
            # Trimming the data so that the sample size is consistent across subjects
            if root=='./MIdata/BNCI2014004':
                data['X'] = data['X'][:680]
                data['y'] = data['y'][:680]
            if root=='./MIdata/BNCI2015001':
                data['X'] = data['X'][:400]
                data['y'] = data['y'][:400]
            if root=='./MIdata/BNCI2015001_rest':
                data['X'] = data['X'][:400]
                data['y'] = data['y'][:400]    
            N = data['X'].shape[0]
            C = data['X'].shape[1]
            T = data['X'].shape[2]        
            X.append(data['X'])
            df = pd.get_dummies(data['y'])
            y.append(df.to_numpy()) 
            # y.append(data['y']) # Gauss

        self.data = np.array(X) #(subject,N,H,W)
        self.label = np.array(y)
        self.label = self.label.reshape(N*len(self.subject_id),-1)
        #transform
        self.data = self.data_transform(self.data)
        self.label = self.label_transform(self.label)
        self.data = self.data.reshape(N*len(self.subject_id),1,C,T)

        X_train, X_val, y_train, y_val = train_test_split(self.data, self.label, test_size=self.test_size, random_state=self.random_state)

        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val

        # for all
        _ = list(zip(self.data, self.label))
        np.random.seed(self.random_state)
        np.random.shuffle(_)
        self.data, self.label = zip(*_) #tuple

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.X_train[index], self.y_train[index]
        elif self.mode == 'val':
            return self.X_val[index], self.y_val[index]
        elif self.mode == 'all':
            return self.data[index], self.label[index]

    def __len__(self):
        if self.mode == 'train':
            return len(self.X_train)
        elif self.mode == 'val':
            return len(self.X_val)
        elif self.mode == 'all':
            return len(self.data)


if __name__ == '__main__':
    data_transform = [
        # EA(),
        ArrayToTensor()
    ]
    label_transform = [
        ArrayToTensor()
    ] 
    data = MIDataset(random_state=42, subject_id=[1,2,3,4,5,6,7,8], root='../data/BNCI2014008', mode='all',data_transform=data_transform, label_transform=label_transform)
    print(data.X_train.shape)
    print(type(data.X_train))
    print(data.y_val.shape)
    # print(data.y_val)
    # print(data.input_shape)
    # print(data.data.size())