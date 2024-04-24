import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

from model import *
from utils import *
from SAM import *
from otherFedComponents import *

class Client(object):
    def __init__(self, args, train_dataset, eval_dataset, id=-1):
        self.args = args
        bn_track = False if args.fedbs else True
        if self.args.model == 'eegnet':
            if self.args.fedfa:
                self.local_model = EEGNetFedFa(sample_rate=args.sample_rate, channels=args.channels, F1=args.F1, D=args.D, F2=args.F2, 
                    time=args.samples, class_num=args.class_num, drop_out=args.dropout, bn_track=bn_track, prob=args.prob)
            else:
                self.local_model = EEGNet(sample_rate=args.sample_rate, channels=args.channels, F1=args.F1, D=args.D, F2=args.F2, 
                    time=args.samples, class_num=args.class_num, drop_out=args.dropout, bn_track=bn_track)
        elif self.args.model == 'deepconvnet':
            if self.args.fedfa:
                self.local_model = DeepConvNetFa(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times, prob=args.prob)
            else: 
                self.local_model = DeepConvNet(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times)
        elif self.args.model == 'shallowconvnet':
            if self.args.fedfa:
                self.local_model = ShallowConvNetFa(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times, prob=args.prob)
            else: 
                self.local_model = ShallowConvNet(n_classes=args.class_num, Chans=args.channels, Samples=args.samples, 
                    dropoutRate=args.dropout, bn_track=bn_track, TemporalKernel_Times=args.TemporalKernel_Times)
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.client_id = id
        self.local_model = self.local_model.to(self.device)

        self.local_model.apply(weights_init) 

        if self.args.scaffold:
            self.c_local = [torch.zeros_like(param).to(self.device) for param in self.local_model.parameters()]
            self.c_diff = []
        
        if self.args.moon:
            self.previous_model = copy.deepcopy(self.local_model).to(self.device)

    def local_train(self, model,c_global=None):

        if self.args.scaffold:
            for c_l, c_g in zip(self.c_local, c_global):
                self.c_diff.append(-c_l+c_g)
        
        if self.args.moon:
            self.previous_model = copy.deepcopy(self.local_model).to(self.device)
            cos = torch.nn.CosineSimilarity(dim=-1)
            featurizer_global = torch.nn.Sequential(*(list(model.children())[:-1]))
            featurizer_previous = torch.nn.Sequential(*(list(self.previous_model.children())[:-1]))

        # fedbs does not distribute bn layer statistic parameters
        if self.args.fedbs:
            for name, param in model.state_dict().items():
                if 'bn' not in name and 'running_mean_bmic' not in name and 'running_std_bmic' not in name: 
                    self.local_model.state_dict()[name].copy_(param.clone())
        else:
            for name, param in model.state_dict().items():
                if 'running_mean_bmic' not in name and 'running_std_bmic' not in name:
                    self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.args.lr, weight_decay=1e-4, momentum=0.9)

        for i in range(self.args.local_epochs):
            train_loss = 0
            train_acc = 0

            for batch_idx, (X,y) in enumerate(self.train_dataloader):
                # fedbs uses SAM for local optimization
                if self.args.fedbs:
                    minimizer = SAM(optimizer, self.local_model, self.args.rho)
                    X,y = X.to(self.device), y.to(self.device)
                    
                    # Ascent Step
                    y_hat = self.local_model(X)
                    loss = criterion(y_hat,y)
                    loss.backward()
                    minimizer.ascent_step()

                    # Descent Step
                    criterion(self.local_model(X),y).backward()
                    minimizer.descent_step()
            
                else:
                    X,y = X.to(self.device), y.to(self.device)

                    y_hat = self.local_model(X)
                    loss = criterion(y_hat,y)

                    # MOON Contrastive loss
                    if self.args.moon:
                        featurizer = torch.nn.Sequential(*(list(self.local_model.children())[:-1]))
                        with torch.no_grad():
                            projection_global = featurizer_global(X).view(X.size(0),-1)
                            projection_previous = featurizer_previous(X).view(X.size(0),-1)
                        projection = featurizer(X).view(X.size(0), -1)
                        posi = cos(projection,projection_global)
                        nega = cos(projection,projection_previous)
                        embeddings_contrastive = torch.cat((posi.reshape(-1,1),nega.reshape(-1,1)),dim=1)
                        embeddings_contrastive /= self.args.temperature
                        labels_contrastive = torch.zeros(X.size(0)).to(self.device).long()
                        contrastive_loss = self.args.mu_moon * criterion(embeddings_contrastive, labels_contrastive)
                        loss += contrastive_loss


                    # FedProx proximal loss
                    if self.args.fedprox:
                        proximal_loss = 0
                        # if self.args.mu != 0:
                        #     for name,data in self.local_model.state_dict().items(): 
                        #         proximal_loss = proximal_loss+(proximal(data.float(),model.state_dict()[name].float()))
                        for w_s, w_c in zip(model.parameters(),self.local_model.parameters()): # w/o non-training parameters
                            proximal_loss += torch.pow(torch.norm(w_s-w_c),2)
                        loss += (self.args.mu/2.) * proximal_loss
                        
                    optimizer.zero_grad()
                    loss.backward()

                    if self.args.scaffold:
                        for param, c_d in zip(self.local_model.parameters(), self.c_diff):
                            param.grad += c_d.data

                    
                    optimizer.step()
                    # model.MaxNormConstraint()

                train_loss += loss
                pred = y_hat.max(-1, keepdim=True)[1]
                y_true = y.max(-1, keepdim=True)[1]
                train_acc += pred.eq(y_true).sum().item()
            
            train_loss = train_loss / len(self.train_dataloader.dataset)
            train_acc = train_acc / len(self.train_dataloader.dataset) 

        # Calculate client gradient
        weight_diff = dict()
        for name, data in self.local_model.state_dict().items():
            weight_diff[name] = (data-model.state_dict()[name])

        if self.args.scaffold:
            c_plus = []
            c_delta = []
            # compute c_plus
            coef = 1/(self.args.lr*self.args.local_epochs*len(self.train_dataloader))
            for c_l, c_g, param_g, param_l in zip(self.c_local, c_global, model.parameters(), self.local_model.parameters()):
                c_plus.append(c_l-c_g+coef*(param_g-param_l))          
            # compute c_delta
            for c_p,c_l in zip(c_plus, self.c_local):
                c_delta.append(c_p-c_l)

            return weight_diff, c_delta
        
        elif self.args.fedfa:
            client_running_mean = {}
            client_running_std = {}

            for name,data in self.local_model.state_dict().items():
                if "running_mean_bmic" in name:
                    client_running_mean[name] = data
                if "running_std_bmic" in name:
                    client_running_std[name] = data

            return weight_diff, client_running_mean, client_running_std


        return weight_diff

    def local_eval(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
            
        self.local_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        eval_acc = 0
        eval_loss = 0
        for batch_idx, (X,y) in enumerate(self.eval_dataloader):
            X,y = X.to(self.device), y.to(self.device)

            with torch.no_grad():
                y_hat = self.local_model(X)

            loss = criterion(y_hat, y)
            pred = y_hat.max(-1, keepdim=True)[1]
            y_true = y.max(-1, keepdim=True)[1]

            eval_acc += pred.eq(y_true).sum().item()
            eval_loss += loss

        eval_loss = eval_loss/len(self.eval_dataloader.dataset)
        eval_acc = eval_acc/len(self.eval_dataloader.dataset)

        return eval_loss, eval_acc
        


