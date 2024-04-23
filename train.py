import argparse
import json
import random
from datetime import datetime
import time

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
from weight_adjust_GA import *

def train(conf, server_subject_id, client_subject_id, allserver_test_acc_list, trace_func=print, save_path = './checkpoint.pth'):
    # Data
    seed = random.randint(1,100) 
    data_transform = [
        # EA(),
        # ZScoreNorm(),
        ArrayToTensor()
    ]
    label_transform = [
        ArrayToTensor()
    ]
    test_dataset = MIDataset(random_state=seed,subject_id=server_subject_id, root=conf['data_path'],
                                        mode='all', data_transform=data_transform, label_transform=label_transform)

    
    #---------------------------------------------------# 
    client_acc_set = {}
    client_loss_set = {}
    client_epoch_set = {}
    server_loss_list = []
    server_acc_list = []
    server_epoch_list = []

    ## DG-GA
    clientloss_after_avg = {}
    clientacc_after_avg = {}
    clientloss_before_avg = {}  
    clientacc_before_avg = {} 
    step_size = conf['step_size']
    step_size_decay = step_size / conf['global_epochs']                  

    for i in client_subject_id:
        client_acc_set['client'+str(i)] = []
        client_loss_set['client'+str(i)] = []
        client_epoch_set['client'+str(i)] = []
    #---------------------------------------------------#

    
    early_stopping = EarlyStopping(patience=conf['patience'], verbose=False, delta=0, path=save_path, trace_func=trace_func, counter_info=False, is_save=True, early=conf['early'])

    # Server & Client Init
    trace_func(f'Begin Initing Server {server_subject_id} & Clients {client_subject_id}')
    server = Server(conf, test_dataset)
    clients = []
    for i in client_subject_id:
        id = []
        id.append(i) # list
        clients.append(Client(conf,MIDataset(random_state=seed, subject_id=id, root=conf['data_path'], 
                                mode='all', data_transform=data_transform, label_transform=label_transform),
                                MIDataset(random_state=seed, subject_id=id, root=conf['data_path'], 
                                mode='all', data_transform=data_transform, label_transform=label_transform), id=i))

        
    trace_func(f'Begin Training, Server subject id: {server_subject_id}')
    for epoch in range(conf['global_epochs']):
        candidates = random.sample(clients, conf['sample_num'])
        candidates_id_list = [j.client_id for j in candidates]
        if epoch == 0: 
            avg_weight_dict = {}
            for id in client_subject_id:
                avg_weight_dict[id] = 1/len(candidates_id_list)
                clientloss_before_avg[id] = None
                clientloss_after_avg[id] = None

        if conf['scaffold']:
            c_delta_list = []
        if conf['fedfa']:
            client_running_mean_list = []
            client_running_std_list = []

        eval_loss = 0
        eval_acc = 0
        valid_nums = 0

        client_weight_dict = {} 
        for j in candidates:
            weight_accumulator = {}
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)
            client_weight_dict[j.client_id] = weight_accumulator

        for j in candidates:
            if conf['scaffold']:
                weight_diff, valid, c_delta = j.local_train(server.global_model, client_acc_set['client'+str(j.client_id)], client_loss_set['client'+str(j.client_id)], 
                            client_epoch_set['client'+str(j.client_id)], epoch, server.c_global)
                c_delta_list.append(c_delta)
            elif conf['fedfa']:
                weight_diff, valid, client_running_mean, client_running_std = j.local_train(server.global_model, client_acc_set['client'+str(j.client_id)], client_loss_set['client'+str(j.client_id)], 
                            client_epoch_set['client'+str(j.client_id)], epoch)
                client_running_mean_list.append(client_running_mean)
                client_running_std_list.append(client_running_std)
            else:
                # clientloss_after_lastavg[j.client_id-1], clientacc_after_lastavg[j.client_id-1] = j.local_eval(server.global_model)
                weight_diff, valid = j.local_train(server.global_model, client_acc_set['client'+str(j.client_id)], client_loss_set['client'+str(j.client_id)], 
                            client_epoch_set['client'+str(j.client_id)], epoch)
                # clientloss_before_avg[j.client_id-1],clientacc_before_avg[j.client_id-1] = j.local_eval(j.local_model)
                clientloss_before_avg[j.client_id], _ = j.local_eval(j.local_model)
                clientloss_after_avg[j.client_id], _ = j.local_eval(server.global_model) 
            if valid:
                valid_nums += 1
                # evaluation
                loss, acc = j.local_eval(server.global_model)
                eval_loss = eval_loss + loss
                eval_acc = eval_acc + acc
                # weight accumalate
                for name, params in server.global_model.state_dict().items():
                    client_weight_dict[j.client_id][name].add_(weight_diff[name]) 

        eval_loss = eval_loss/valid_nums if valid_nums!=0 else float('inf')
        eval_acc = eval_acc/valid_nums if valid_nums!=0 else 0

        
        early_stopping(eval_loss,server.global_model,eval_acc,epoch, conf['global_epochs'])

        
        if valid_nums != 0:
            if conf['scaffold']:
                server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list, c_delta_list=c_delta_list)
            elif conf['fedfa']:
                server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list, client_running_mean_list=client_running_mean_list, client_running_std_list=client_running_std_list)
            else:
                # if epoch == 0:
                #     temp_clientloss_before_avg = clientloss_before_avg
                # else: 
                #     temp_clientloss_after_avg = clientloss_after_lastavg
                #     avg_weight_list = refine_weight_dict_by_GA(avg_weight_list, temp_clientloss_before_avg, temp_clientloss_after_avg, step_size=0.1, fair_metric='loss')
                #     temp_clientloss_before_avg = clientloss_before_avg
                server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list)
                if conf['GA']:
                    avg_weight_dict = refine_weight_dict_by_GA(avg_weight_dict, candidates_id_list, clientloss_before_avg, clientloss_after_avg, step_size=step_size-epoch*step_size_decay, fair_metric='loss')

            #-------------------------------------------#
            server_loss_list.append(eval_loss.item())
            server_acc_list.append(100*eval_acc)
            server_epoch_list.append(epoch)
            #-------------------------------------------#

        if early_stopping.early_stop:
            trace_func(f'Server Subject {server_subject_id} 早停, best/final epoch:{early_stopping.best_epoch}')
            trace_func(f'Best/Final Val Acc: {100*early_stopping.best_val_acc:.2f}%')
            break
    
    if not early_stopping.early_stop:
        trace_func(f'Server Subject {server_subject_id} 未早停, best/final epoch:{early_stopping.best_epoch}')
        trace_func(f'Best/Final Val Acc: {100*early_stopping.best_val_acc:.2f}%')

    # Test
    server.global_model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = server.model_test()
    allserver_test_acc_list.append(round(100*test_acc,2))
    trace_func(f'Final Test Acc: {100*test_acc:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--conf_path', type=str, default='./conf_eegnet_BNCI2014001.json', help='path to the conf')
    parser.add_argument('--nums', type=int, default=1, help='the number of run times')

    args = parser.parse_args()

    with open(args.conf_path, 'r') as f:
        conf = json.load(f)
    print(conf)

    # Create multi_results folders
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print('timestamp: ', timestamp)
    multi_result_path = '%s/multi_results/%s'%(conf['output_path'], timestamp)
    save_path = '%s/save_models/%s'%(conf['output_path'], timestamp)
    os.makedirs(multi_result_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    nums = args.nums 
    total_acc_list = []
    total_index = []
    print('总体运行次数: ', nums)
    print()

    start_time = time.time()
    print("开始时间: ", start_time)
    for num in range(1,nums+1):
        print('=============================================================================================')
        print(f'开始第{num}次训练')
        
        allserver_test_acc_list = []
        subject_id = [int(i) for i in conf['sub_id'].split(',')]

        for id in subject_id:
            server_subject_id = []
            server_subject_id.append(id)
            tmp = subject_id.copy()
            tmp.remove(id)
            client_subject_id = tmp

            print('----------------------------------------------------------------')
            print(f'Server Subject {server_subject_id} Train Begin')
            print('This time server subject ID: ',server_subject_id)
            print('This time client subject ID: ',client_subject_id)
            train(conf,server_subject_id,client_subject_id,allserver_test_acc_list, trace_func=tqdm.write, save_path='%s/Model_ServerSub%s.pth'%(save_path,str(id)))
            print(f'Server Subject {server_subject_id} Train Complete')
            print('----------------------------------------------------------------')

        mean = round(sum(allserver_test_acc_list)/len(allserver_test_acc_list),2)
        allserver_test_acc_list.append(mean)

        total_acc_list.append(allserver_test_acc_list)
        print(f'第{num}次训练完成')
        print('==============================================================\n')

    end_time = time.time()
    print("结束时间: ",end_time)
    training_time = end_time-start_time
    print("训练总共所花费的时间：{} 秒".format(training_time))
    print("平均到一次的训练时间为: {} 秒".format(training_time/len(subject_id)))

    for num in range(1,nums+1):
        total_index.append(f'Time {num}')

    total_columns = [int(i) for i in conf['sub_id'].split(',')]
    total_columns.append('Avg')
    df = pd.DataFrame(total_acc_list, columns=total_columns, index=total_index)
    print('最终Test Acc Table如下:')
    print(df)

    df.to_excel(f'{multi_result_path}/{timestamp}_multi_test_acc.xlsx')

