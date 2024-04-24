import numpy as np
import sys
sys.path.append('.')

from moabb.datasets import BNCI2014001,BNCI2014002,BNCI2015001
from moabb.paradigms import MotorImagery
import scipy.io as scio

def download_data(MI, resample=250):
    if MI == 'MI1':
        root="./BNCI2014001/" 
        dataset = BNCI2014001()
        n_classes = 4
    elif MI == 'MI2':
        root="./BNCI2014002/" 
        dataset = BNCI2014002()
        n_classes = 2
    elif MI == 'MI3':
        root="./BNCI2015001/" 
        dataset = BNCI2015001()
        n_classes = 2
    tail=".mat"

    subjects = dataset.subject_list
    print('Subject ', len(subjects))
    for i in range(len(subjects)):
        for j in range(1): # session
            print('Downloading subject ',subjects[i],' data')
            print('session',j)
            paradigm = MotorImagery(n_classes=n_classes, fmin=8, fmax=30, resample=resample) 
            X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subjects[i]])
            print('y',y.shape) 
            print('X',X.shape) 
        filename = root + str(i+1) + tail

        scio.savemat(filename, {'X':X,'y': y}) 

if __name__ == '__main__':
    download_data('MI1')
    download_data('MI2')
    download_data('MI3')