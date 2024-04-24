import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import *
from utils import *
from otherFedComponents import *

class Server(object):
    def __init__(self, args, test_dataset):
        self.args = args
        bn_track = False if args.fedbs else True
        if self.args.model == 'eegnet':
            if self.args.fedfa:
                self.global_model = EEGNetFedFa(sample_rate=args.sample_rate, channels=args.channels, F1=args.F1, D=args.D, F2=args.F2, 
                    time=args.samples, class_num=args.class_num, drop_out=args.dropout, bn_track=bn_track, prob=args.prob)
            else:
                self.global_model = EEGNet(sample_rate=args.sample_rate, channels=args.channels, F1=args.F1, D=args.D, F2=args.F2, 
                    time=args.samples, class_num=args.class_num, drop_out=args.dropout, bn_track=bn_track)
        elif self.args.model == 'deepconvnet':
            if self.args.fedfa:
                self.global_model = DeepConvNetFa(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times, prob=args.prob)
            else: 
                self.global_model = DeepConvNet(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times)
        elif self.args.model == 'shallowconvnet':
            if self.args.fedfa:
                self.global_model = ShallowConvNetFa(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times, prob=args.prob)
            else: 
                self.global_model = ShallowConvNet(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times)

        self.test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = self.global_model.to(self.device)

        self.global_model.apply(weights_init) 

        if self.args.scaffold:
            self.c_global = [torch.zeros_like(param).to(self.device) for param in self.global_model.parameters()]

    def model_aggregate(self, client_weight_dict, avg_weight_dict, candidates_id_list, c_delta_list=None, client_running_mean_list=None, client_running_std_list=None):
        for id in candidates_id_list:
            for name, data in self.global_model.state_dict().items():
                update_per_layer = client_weight_dict[id][name]*avg_weight_dict[id]
                if data.type() != update_per_layer.type():
                    data.add_(update_per_layer.to(torch.int64))  
                else:
                    data.add_(update_per_layer)

        if self.args.scaffold:
            avg_weight = torch.tensor(
            [
                1 / self.args.sample_num
                for _ in range(self.args.sample_num)
            ],
            device=self.device,
            )
            for c_g, c_del in zip(self.c_global, zip(*c_delta_list)):
                c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
                c_g.data += (self.args.sample_num/(len([int(i) for i in self.args.sub_id.split(',')])-1))*c_del

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
    