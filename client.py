import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

from model import *
from utils import *
from minimizers import *

class Client(object):
    def __init__(self, conf, train_dataset, eval_dataset, id=-1):
        self.conf = conf
        if self.conf['model'] == 'eegnet':
            if self.conf['fedfa']:
                self.local_model = EEGNetFedFa(sample_rate=conf['sample_rate'], channels=conf['channels'], F1=conf['F1'], D=conf['D'], F2=conf['F2'], 
                    time=conf['time'], class_num=conf['class_num'], drop_out=conf['drop_out'], bn_track=conf['bn_track'], prob=conf['prob'])
            else:
                self.local_model = EEGNet(sample_rate=conf['sample_rate'], channels=conf['channels'], F1=conf['F1'], D=conf['D'], F2=conf['F2'], 
                    time=conf['time'], class_num=conf['class_num'], drop_out=conf['drop_out'], bn_track=conf['bn_track'])
        elif self.conf['model'] == 'deepconvnet':
            if self.conf['fedfa']:
                self.local_model = DeepConvNetFa(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'],prob=conf['prob'])
            else: 
                self.local_model = DeepConvNet(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'])
        elif self.conf['model'] == 'shallowconvnet':
            if self.conf['fedfa']:
                self.local_model = ShallowConvNetFa(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'],prob=conf['prob'])
            else: 
                self.local_model = ShallowConvNet(n_classes=conf['n_classes'], Chans=conf['Chans'], Samples=conf['Samples'], 
                    dropoutRate=conf['dropoutRate'], bn_track=conf['bn_track'], TemporalKernel_Times=conf['TemporalKernel_Times'])
        self.train_dataloader = DataLoader(train_dataset, batch_size=conf['batch_size'], shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=conf['batch_size'], shuffle=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.client_id = id
        self.local_model = self.local_model.to(self.device)

        self.local_model.apply(weights_init) 

        if self.conf['scaffold']:
            self.c_local = [torch.zeros_like(param).to(self.device) for param in self.local_model.parameters()]
            self.c_diff = []
        
        if self.conf['moon']:
            self.previous_model = copy.deepcopy(self.local_model).to(self.device)

    def local_train(self, model, client_acc_list, client_loss_list, client_epoch_list, complete_global_epoch,c_global=None):
        complete_local_epoch = self.conf['local_epochs'] * complete_global_epoch

        if self.conf['scaffold']:
            for c_l, c_g in zip(self.c_local, c_global):
                self.c_diff.append(-c_l+c_g)
        
        if self.conf['moon']:
            self.previous_model = copy.deepcopy(self.local_model).to(self.device)
            cos = torch.nn.CosineSimilarity(dim=-1)
            featurizer_global = torch.nn.Sequential(*(list(model.children())[:-1]))
            featurizer_previous = torch.nn.Sequential(*(list(self.previous_model.children())[:-1]))

        if self.conf['fedbn']:
            for name, param in model.state_dict().items():
                if 'bn' not in name and 'running_mean_bmic' not in name and 'running_std_bmic' not in name: 
                    self.local_model.state_dict()[name].copy_(param.clone())
        else:
            for name, param in model.state_dict().items():
                if 'running_mean_bmic' not in name and 'running_std_bmic' not in name:
                    self.local_model.state_dict()[name].copy_(param.clone())
                # self.local_model.state_dict()[name].copy_(param.clone())

        self.local_model.train()
        criterion = nn.CrossEntropyLoss()
        # proximal = nn.MSELoss() #FedProx proximal loss
        if self.conf['decay'] == True:
            # lr = self.conf['lr'] * np.exp((-1)*self.conf['decay_rate']*complete_global_epoch)
            lr = self.conf['lr'] * np.power(self.conf['decay_rate'], complete_global_epoch)
        else:
            lr = self.conf['lr']
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
        # optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

        for i in range(self.conf['local_epochs']):
            train_loss = 0
            train_acc = 0

            for batch_idx, (X,y) in enumerate(self.train_dataloader):
                if self.conf['sam'] == 'no':
                    X,y = X.to(self.device), y.to(self.device)

                    y_hat = self.local_model(X)
                    loss = criterion(y_hat,y)

                    # MOON Contrastive loss
                    if self.conf['moon']:
                        featurizer = torch.nn.Sequential(*(list(self.local_model.children())[:-1]))
                        with torch.no_grad():
                            projection_global = featurizer_global(X).view(X.size(0),-1)
                            projection_previous = featurizer_previous(X).view(X.size(0),-1)
                        projection = featurizer(X).view(X.size(0), -1)
                        posi = cos(projection,projection_global)
                        nega = cos(projection,projection_previous)
                        embeddings_contrastive = torch.cat((posi.reshape(-1,1),nega.reshape(-1,1)),dim=1)
                        embeddings_contrastive /= self.conf['temperature']
                        labels_contrastive = torch.zeros(X.size(0)).to(self.device).long()
                        contrastive_loss = self.conf['mu_moon'] * criterion(embeddings_contrastive, labels_contrastive)
                        loss += contrastive_loss


                    # FedProx proximal loss
                    if self.conf['mu'] != 0:
                        proximal_loss = 0
                        # if self.conf['mu'] != 0:
                        #     for name,data in self.local_model.state_dict().items(): 
                        #         proximal_loss = proximal_loss+(proximal(data.float(),model.state_dict()[name].float()))
                        for w_s, w_c in zip(model.parameters(),self.local_model.parameters()): # w/o non-training parameters
                            proximal_loss += torch.pow(torch.norm(w_s-w_c),2)
                        loss += (self.conf['mu']/2.) * proximal_loss

                    optimizer.zero_grad()
                    loss.backward()

                    if self.conf['scaffold']:
                        if self.conf['fedbn']:
                            for (name, param), c_d in zip(self.local_model.named_parameters(), self.c_diff):
                                if 'bn' not in name:
                                    param.grad += c_d.data
                        else: 
                            for param, c_d in zip(self.local_model.parameters(), self.c_diff):
                                param.grad += c_d.data

                    optimizer.step()
                    # model.MaxNormConstraint()
                elif self.conf['sam'] in ['sam','asam']:
                    minimizer = SAM(optimizer, self.local_model, self.conf['rho'], self.conf['eta']) if self.conf['sam'] == 'sam' else ASAM(optimizer, self.local_model, self.conf['rho'], self.conf['eta'])
                    X,y = X.to(self.device), y.to(self.device)
                    
                    # Ascent Step
                    y_hat = self.local_model(X)
                    loss = criterion(y_hat,y)
                    loss.backward()
                    minimizer.ascent_step()

                    # Descent Step
                    criterion(self.local_model(X),y).backward()
                    minimizer.descent_step()

                train_loss += loss
                pred = y_hat.max(-1, keepdim=True)[1]
                y_true = y.max(-1, keepdim=True)[1]
                train_acc += pred.eq(y_true).sum().item()
            
            train_loss = train_loss / len(self.train_dataloader.dataset)
            train_acc = train_acc / len(self.train_dataloader.dataset) 
            client_acc_list.append(100*train_acc)
            client_loss_list.append(train_loss.item())
            client_epoch_list.append(complete_local_epoch+i+1)

        weight_diff = dict()
        for name, data in self.local_model.state_dict().items():
            weight_diff[name] = (data-model.state_dict()[name])
        
        if complete_global_epoch < self.conf['warmup_epoch']:
            valid = True
        else:
            valid = True if train_acc >= self.conf['threshold'] else False

        if self.conf['scaffold']:
            c_plus = []
            c_delta = []
            # compute c_plus
            coef = 1/(lr*self.conf['local_epochs']*len(self.train_dataloader))
            for c_l, c_g, param_g, param_l in zip(self.c_local, c_global, model.parameters(), self.local_model.parameters()):
                c_plus.append(c_l-c_g+coef*(param_g-param_l))          
            # compute c_delta
            for c_p,c_l in zip(c_plus, self.c_local):
                c_delta.append(c_p-c_l)

            return weight_diff, valid, c_delta
        
        elif self.conf['fedfa']:
            client_running_mean = {}
            client_running_std = {}

            for name,data in self.local_model.state_dict().items():
                if "running_mean_bmic" in name:
                    client_running_mean[name] = data
                if "running_std_bmic" in name:
                    client_running_std[name] = data

            return weight_diff, valid, client_running_mean, client_running_std


        return weight_diff, valid

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
        

         

