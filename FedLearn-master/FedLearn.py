from typing import Callable, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from Utils import BasicDataset, copy_model_first_k_layers_params, copy_model_params


class FederatedScheme:
    def __init__(self, clients: List[BasicDataset], create_model: Callable[[], nn.Module], loss_func: Callable,
                 device: str='cuda:0',drop_rate: float=0.5):
        self.clients = clients
        self.create_model = create_model
        self.global_model = create_model().to(device)
        self.loss_func = loss_func
        self.device = device
        print("device",self.device)
        # The communication rounds
        self.comm_rounds = 0
        self.comm_size = 0

        # use to implement sparse update
        self.dropping_rate=drop_rate
        self.residuals = {}

        # use to implement fed-per algorithm
        self.kp=1
        self.total=4 if create_model.__name__ == '3NN' else 3
        self.client_models= {}

    def add_client_dateset(self, data: np.ndarray):
        current_client_id = len(self.clients)
        self.clients.append(BasicDataset(data))
        return current_client_id

    def train_client(self, client_id: int, batch_size: int, n_batches: int, learning_rate: float):
        # Copy the global model
        local_model = self.create_model().to(self.device)
        copy_model_params(self.global_model, local_model)
        optimizer = SGD(local_model.parameters(), learning_rate)

        # Train the local model
        for i in range(n_batches):
            batch_xs, batch_ys = self.clients[client_id].get_batch_xy(batch_size)
            batch_xs = torch.from_numpy(batch_xs).to(self.device)
            batch_ys = torch.from_numpy(batch_ys).to(self.device)
            optimizer.zero_grad()
            loss = self.loss_func(local_model(batch_xs), batch_ys)
            # print(loss.item())
            loss.backward()
            optimizer.step()

        # Return the local model
        return local_model
    
    def train_client_sparse(self, client_id: int, batch_size: int, n_batches: int, learning_rate: float):
        local_model = self.create_model().to(self.device)
        if client_id in self.residuals:
            residuals = self.residuals[client_id]
        else :
            residuals = [torch.zeros_like(param) for param in local_model.parameters()]
        # Copy the global model
        copy_model_params(self.global_model, local_model)
        optimizer = SGD(local_model.parameters(), learning_rate)

        # Train the local model
        for i in range(n_batches):
            batch_xs, batch_ys = self.clients[client_id].get_batch_xy(batch_size)
            batch_xs = torch.from_numpy(batch_xs).to(self.device)
            batch_ys = torch.from_numpy(batch_ys).to(self.device)
            optimizer.zero_grad()
            loss = self.loss_func(local_model(batch_xs), batch_ys)
            # print(loss.item())
            loss.backward()
            optimizer.step()
        # sparse update
        # Delta is a tensor of the minus of the gradients between the global model and the local model
        Delta = [param.data - global_param.data for param, global_param in zip(local_model.parameters(), self.global_model.parameters())]            
        Delta = [delta + residual for delta, residual in zip(Delta, residuals)]
        Dropped = [torch.zeros_like(delta) for delta in Delta]
        # calculate the threshold for each tensor
        for i, delta in enumerate(Delta):
            # threshold is the top k value of the absolute value of the tensor
            indice=int(delta.numel() * self.dropping_rate)
            threshold=torch.sort(torch.abs(delta.view(-1)))[0][indice]
            # set the value of the tensor to Dropped if it is larger than the threshold
            Dropped[i][torch.abs(delta) > threshold] = delta[torch.abs(delta) > threshold]
        # update the residual
        self.residuals[client_id] = [delta - dropped for delta, dropped in zip(Delta, Dropped)]
        # update the local model
        for param, dropped in zip(self.global_model.parameters(), Dropped):
            param.data = param.data + dropped
        return local_model

    def fed_avg_one_step(self, n_clients: int, local_batch_size: int, n_local_batches: int, local_learning_rate: float):
        client_indices = np.random.choice(len(self.clients), n_clients)
        local_paras = [[] for _ in self.global_model.parameters()]
        for i, client_idx in enumerate(client_indices):
            client_model = self.train_client_sparse(client_idx, local_batch_size, n_local_batches, local_learning_rate)
            for j, param in enumerate(client_model.parameters()):
                local_paras[j].append(param.data)

        # Set the global weights to be mean of local weights
        for i, param in enumerate(self.global_model.parameters()):
            param.data = torch.mean(torch.stack(local_paras[i]), dim=0)

        self.comm_rounds += 1
        self.comm_size += n_clients

    #客户端损失占比来决定汇总的参数占比：
    def fed_avg_one_step_loss(self, n_clients: int, local_batch_size: int, n_local_batches: int, local_learning_rate: float):
        client_indices = np.random.choice(len(self.clients), n_clients)
        local_paras = [[] for _ in self.global_model.parameters()]
        local_loss = []
        for i, client_idx in enumerate(client_indices):
            client_model,client_loss = self.train_client(client_idx, local_batch_size, n_local_batches, local_learning_rate)
            local_loss.append(client_loss)
            for j, param in enumerate(client_model.parameters()):
                local_paras[j].append(param.data)
                
        # Set the global weights to be mean of local weights
        for i, param in enumerate(self.global_model.parameters()):
            x=local_loss[i]/sum(local_loss)
            param.data = torch.sum(torch.stack(local_paras[i]), dim=0)*local_loss[i]/sum(local_loss)
            
        self.comm_rounds += 1
        self.comm_size += n_clients
    
    def train_client_fedper(self, client_id: int, batch_size: int, n_batches: int, learning_rate: float):
        if client_id in self.client_models:
            local_model = self.client_models[client_id]
        else:
            local_model = self.create_model().to(self.device)
        # Copy the global model
        copy_model_first_k_layers_params(self.global_model, local_model,self.total-self.kp)
        optimizer = SGD(local_model.parameters(), learning_rate)

        # Train the local model
        for i in range(n_batches):
            batch_xs, batch_ys = self.clients[client_id].get_batch_xy(batch_size)
            batch_xs = torch.from_numpy(batch_xs).to(self.device)
            batch_ys = torch.from_numpy(batch_ys).to(self.device)
            optimizer.zero_grad()
            loss = self.loss_func(local_model(batch_xs), batch_ys)
            # print(loss.item())
            loss.backward()
            optimizer.step()

        self.client_models[client_id] = local_model
        # Return the local model
        return local_model




    def fed_per_one_step(self, n_clients: int, local_batch_size: int, n_local_batches: int, local_learning_rate: float):
        client_indices = np.random.choice(len(self.clients), n_clients)
        local_paras = [[] for _ in self.global_model.parameters()]
        for i, client_idx in enumerate(client_indices):
            client_model = self.train_client_fedper(client_idx, local_batch_size, n_local_batches, local_learning_rate)
            for j, param in enumerate(client_model.parameters()):
                local_paras[j].append(param.data)

        # Set the global weights to be mean of local weights
        cnt=0
        for i, param in enumerate(self.global_model.parameters()):
            param.data = torch.mean(torch.stack(local_paras[i]), dim=0)
            cnt+=1
            if cnt==self.total-self.kp:
                break

        self.comm_rounds += 1
        self.comm_size += n_clients