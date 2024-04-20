import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import *
from utils import *

class Server(object):
    def __init__(self, conf, test_dataset):
        self.conf = conf
        if self.conf['model'] == 'eegnet':
            if self.conf['fedfa']:
                self.global_model = EEGNetFedFa(sample_rate=conf['sample_rate'], channels=conf['channels'], F1=conf['F1'], D=conf['D'], F2=conf['F2'], 
                    time=conf['time'], class_num=conf['class_num'], drop_out=conf['drop_out'], bn_track=conf['bn_track'], prob=conf['prob'])
            else:
                self.global_model = EEGNet(sample_rate=conf['sample_rate'], channels=conf['channels'], F1=conf['F1'], D=conf['D'], F2=conf['F2'], 
                    time=conf['time'], class_num=conf['class_num'], drop_out=conf['drop_out'], bn_track=conf['bn_track'])
        elif self.conf['model'] == 'deepconvnet':
            if self.conf['fedfa']:
                self.global_model = DeepConvNetFa(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'],prob=conf['prob'])
            else: 
                self.global_model = DeepConvNet(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'])
        elif self.conf['model'] == 'shallowconvnet':
            if self.conf['fedfa']:
                self.global_model = ShallowConvNetFa(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'],prob=conf['prob'])
            else: 
                self.global_model = ShallowConvNet(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'])
        # self.test_dataloader = DataLoader(test_dataset, batch_size=self.conf['batch_size'], shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = self.global_model.to(self.device)

        self.global_model.apply(weights_init) 

        if self.conf['scaffold']:
            self.c_global = [torch.zeros_like(param).to(self.device) for param in self.global_model.parameters()]

    def model_aggregate(self, client_weight_dict, avg_weight_dict, candidates_id_list, c_delta_list=None, client_running_mean_list=None, client_running_std_list=None):
        # for name, data in self.global_model.state_dict().items():
        #     update_per_layer = weight_accumulator[name]*(1/valid_nums) 
        #     if data.type() != update_per_layer.type():
        #         data.add_(update_per_layer.to(torch.int64))  #为什么是torch.int64 #加上weight_diff
        #     else:
        #         data.add_(update_per_layer)

            
            # if self.conf['fedfa']:
            #     if 'running_var_mean_bmic' in name:
            #         tmp = []
            #         for idx in range(len(client_running_mean_list)):
            #             tmp.append(client_running_mean_list[idx][name.replace('running_var_','running_')])
            #         tmp = torch.stack(tmp)
            #         var = torch.var(tmp)
            #         data.copy_(var)
            #     if 'running_var_std_bmic' in name:
            #         tmp = []
            #         for idx in range(len(client_running_std_list)):
            #             tmp.append(client_running_std_list[idx][name.replace('running_var_','running_')])
            #         tmp = torch.stack(tmp)
            #         var = torch.var(tmp,dim=0)
            #         data.copy_(var)
        for id in candidates_id_list:
            for name, data in self.global_model.state_dict().items():
                update_per_layer = client_weight_dict[id][name]*avg_weight_dict[id]
                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.int64))  
                else:
                    data.add_(update_per_layer)

        if self.conf['scaffold']:
            avg_weight = torch.tensor(
            [
                1 / self.conf['sample_num']
                for _ in range(self.conf['sample_num'])
            ],
            device=self.device,
            )
            for c_g, c_del in zip(self.c_global, zip(*c_delta_list)):
                c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
                c_g.data += (self.conf['sample_num']/(len([int(i) for i in self.conf['sub_id'].split(',')])-1))*c_del

    def model_test(self):
        self.global_model.eval()
        test_acc = 0
        test_loss = 0
        criterion = nn.CrossEntropyLoss()

        for batch_idx, (X,y) in enumerate(self.test_dataloader):
            X,y = X.to(self.device), y.to(self.device)

            with torch.no_grad():
                y_hat = self.global_model(X)

            loss = criterion(y_hat,y)
            test_loss += loss
            pred = y_hat.max(-1, keepdim=True)[1]
            y_true = y.max(-1, keepdim=True)[1]
            test_acc += pred.eq(y_true).sum().item()

        test_loss = test_loss / len(self.test_dataloader.dataset)
        test_acc = test_acc / len(self.test_dataloader.dataset)

        return test_loss, test_acc

    def model_save(self, path):
        torch.save(self.global_model.state_dict(),path)
    