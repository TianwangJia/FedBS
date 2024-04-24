import argparse
import json
import random
from datetime import datetime
import time
import distutils.util

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import pandas as pd

from server import *
from client import *
from datasets import *
from utils import *
from otherFedComponents import *

def train(args, server_subject_id, client_subject_id, Server_TestAcc_List, trace_func=print, save_path = './checkpoint.pth'):
    # Data
    seed = random.randint(1,100) 
    data_transform = [
        EA() if args.ea else None,
        # ZScoreNorm(),
        ArrayToTensor()
    ]
    label_transform = [
        ArrayToTensor()
    ]
    test_dataset = MIDataset(random_state=seed,subject_id=server_subject_id, root=args.data_path,
                                        mode='all', data_transform=data_transform, label_transform=label_transform)
        
    ## DG-GA
    if args.GA:
        clientloss_after_avg = {}
        clientloss_before_avg = {}  
        step_size = args.step_size
        step_size_decay = step_size / args.global_epochs

    
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, delta=0, path=save_path, trace_func=trace_func, counter_info=False, is_save=True, early=args.early)

    # Server & Client Init
    trace_func(f'Begin Initing Server {server_subject_id} & Clients {client_subject_id}')
    server = Server(args, test_dataset)
    clients = []
    for i in client_subject_id:
        id = []
        id.append(i) # list
        # Training set for each client comprises all local data for local training. 
        # While the validation set for each client also comprises of all local data for validate the aggregated global model.
        # Note that the training and validation datasets are consistent , but the models are not.
        # Consistent with the paper, we did not employ global model validation for early stopping. 
        # Instead, we trained for a fixed number of epochs, with validation metrics used solely for monitoring.
        clients.append(Client(args,MIDataset(random_state=seed, subject_id=id, root=args.data_path, 
                                mode='all', data_transform=data_transform, label_transform=label_transform),
                                MIDataset(random_state=seed, subject_id=id, root=args.data_path, 
                                mode='all', data_transform=data_transform, label_transform=label_transform), id=i))

        
    trace_func(f'Begin Training')
    for epoch in range(args.global_epochs):
        candidates = random.sample(clients, args.sample_num)
        candidates_id_list = [j.client_id for j in candidates]
        if epoch == 0: 
            avg_weight_dict = {}
            for id in client_subject_id:
                avg_weight_dict[id] = 1/len(candidates_id_list)
                if args.GA:
                    clientloss_before_avg[id] = None
                    clientloss_after_avg[id] = None

        if args.scaffold:
            c_delta_list = []
        if args.fedfa:
            client_running_mean_list = []
            client_running_std_list = []

        eval_loss = 0
        eval_acc = 0

        client_weight_dict = {} 
        for j in candidates:
            weight_accumulator = {}
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)
            client_weight_dict[j.client_id] = weight_accumulator

        for j in candidates:
            if args.scaffold:
                weight_diff, c_delta = j.local_train(server.global_model, server.c_global)
                c_delta_list.append(c_delta)
            elif args.fedfa:
                weight_diff, client_running_mean, client_running_std = j.local_train(server.global_model)
                client_running_mean_list.append(client_running_mean)
                client_running_std_list.append(client_running_std)
            elif args.GA:
                weight_diff = j.local_train(server.global_model)
                clientloss_before_avg[j.client_id], _ = j.local_eval(j.local_model)
                clientloss_after_avg[j.client_id], _ = j.local_eval(server.global_model) 
            else:
                weight_diff = j.local_train(server.global_model)
                
            # evaluation of the last round global model
            loss, acc = j.local_eval(server.global_model)
            eval_loss = eval_loss + loss
            eval_acc = eval_acc + acc
            # weight accumalate
            for name, params in server.global_model.state_dict().items():
                client_weight_dict[j.client_id][name].add_(weight_diff[name]) 
        
        # average validation metrics of the global model across sampled clients
        eval_loss /= len(candidates)
        eval_acc  /= len(candidates)
        # early stopping is generally not applied
        early_stopping(eval_loss,server.global_model,eval_acc,epoch, args.global_epochs)

        if args.scaffold:
            server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list, c_delta_list=c_delta_list)
        elif args.fedfa:
            server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list, client_running_mean_list=client_running_mean_list, client_running_std_list=client_running_std_list)
        elif args.GA:
            server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list)
            avg_weight_dict = refine_weight_dict_by_GA(avg_weight_dict, candidates_id_list, clientloss_before_avg, clientloss_after_avg, step_size=step_size-epoch*step_size_decay, fair_metric='loss')

        else:
            server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list)

        if early_stopping.early_stop:
            trace_func(f'Stopped early, stop epoch:{early_stopping.best_epoch+1}')
            trace_func(f'Global Model Val Acc: {100*early_stopping.best_val_acc:.2f}%')
            break
    
    if not early_stopping.early_stop:
        trace_func(f'Not stopped early, stop epoch:{early_stopping.best_epoch+1}')
        trace_func(f'Global Model Val Acc: {100*early_stopping.best_val_acc:.2f}%')

    # Test
    server.global_model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = server.model_test()
    Server_TestAcc_List.append(round(100*test_acc,2))
    trace_func(f'Server Test Acc: {100*test_acc:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning for BCIs')

    # About model
    parser.add_argument('--model', type=str, default='eegnet')
    parser.add_argument('--sample_rate', type=int, default=250, help='the sample rate of data')
    parser.add_argument('--F1', type=int, default=8, help='the hyperparameter F1 of eegnet')
    parser.add_argument('--D', type=int, default=2, help='the hyperparameter D if eegnet')
    parser.add_argument('--F2', type=int, default=16, help='the hyperparameter F2 of eegnet')
    parser.add_argument('--class_num', type=int, default=4, help='the classes num of label')
    parser.add_argument('--channels', type=int, default=22, help='the channels num of eeg')
    parser.add_argument('--samples', type=int, default=1001, help='the sampling points in each trial')    
    parser.add_argument('--TemporalKernel_Times', type=int, default=1, help='the times of temporal convolution kernel')
    parser.add_argument('--dropout', type=float, default=0.5, help='the dropout rate of Dropout layer')

    # About basic training setup
    parser.add_argument('--data_path', type=str, default='./data/BNCI2014001', help='path to the datasets')
    parser.add_argument('--sub_id', type=str, default='1,2,3,4,5,6,7,8,9', help='the users of the dataset')
    parser.add_argument('--output_path', type=str, default='./output', help='path to store outputs')
    parser.add_argument('--ea', type=lambda x:bool(distutils.util.strtobool(x)), default=True, help='if true, EA was performed on each subject data')
    
    # About federated training setup
    parser.add_argument('--global_epochs', type=int, default=200, help='the number of global communication rounds')
    parser.add_argument('--sample_num', type=int, default=4, help='the number of clients sampled per round of federal communications')
    parser.add_argument('--local_epochs', type=int, default=2, help='the number of local training epochs on the client')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches of client training')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate of client training')
    parser.add_argument('--early', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='if true, early stopping')
    parser.add_argument('--patience', type=int, default=50, help='the number of communication rounds patience')

    # About setup and hyperparameters for federated approaches
    parser.add_argument('--fedprox', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='if true, perform fedprox')
    parser.add_argument('--mu', type=float, default=1.0, help='mu for fedprox')
    parser.add_argument('--scaffold', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='if true, perform scaffold')
    parser.add_argument('--moon', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='if true, perform moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature for moon')
    parser.add_argument('--mu_moon', type=float, default=1.0, help='mu for moon')
    parser.add_argument('--fedfa', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='if true, perform fedfa')
    parser.add_argument('--prob', type=float, default=0.5, help='probability for fedfa')
    parser.add_argument('--GA', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='if true, perform GA')
    parser.add_argument('--step_size', type=float, default=0.05, help='the step size of GA')
    parser.add_argument('--fedbs', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='if true, perform fedbs')
    parser.add_argument('--rho', type=float, default=0.1, help='rho for fedbs')

    args = parser.parse_args()
    print(args)

    # Create output folders
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('timestamp: ', timestamp)
    save_path = '%s/save_models/%s'%(args.output_path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    print('=============================================================================================')
        
    Server_TestAcc_List = []
    subject_id = [int(i) for i in args.sub_id.split(',')]

    for id in subject_id:
        server_subject_id = []
        server_subject_id.append(id)
        tmp = subject_id.copy()
        tmp.remove(id)
        client_subject_id = tmp

        print('----------------------------------------------------------------')
        print(f'Server Subject {server_subject_id} Begin')
        print('Server subject ID: ',server_subject_id)
        print('Client subject ID: ',client_subject_id)
        train(args,server_subject_id,client_subject_id,Server_TestAcc_List, trace_func=tqdm.write, save_path='%s/Model_ServerSub%s.pth'%(save_path,str(id)))
        print(f'Server Subject {server_subject_id} Complete')
        print('----------------------------------------------------------------')

    mean = round(sum(Server_TestAcc_List)/len(Server_TestAcc_List),2)
    Server_TestAcc_List.append(mean)

    print('==============================================================\n')


    columns = [int(i) for i in args.sub_id.split(',')]
    columns.append('Avg')
    index = ['Test Acc']
    df = pd.DataFrame([Server_TestAcc_List], columns=columns, index=index)
    print('Test Result: ')
    print(df)



