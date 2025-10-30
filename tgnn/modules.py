import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class TimeEncoder(nn.Module):
    def __init__(self,time_dim:int,parameter_requires_grad:bool=True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder,self).__init__()
        self.time_dim=time_dim
        self.w=nn.Linear(1,time_dim)
        self.w.weight=nn.Parameter((torch.from_numpy(1/10**np.linspace(0,9,time_dim,dtype=np.float32))).reshape(time_dim,-1))
        self.w.bias=nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad=False
            self.w.bias.requires_grad=False

    def forward(self,timestamps:torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor,shape (batch_size, seq_len, 1)
        timestamps=timestamps.unsqueeze(dim=2)

        # Tensor,shape (batch_size, seq_len, time_dim)
        output=torch.cos(self.w(timestamps))

        return output

class GAT(MessagePassing):
    def __init__(self,node_dim,latent_dim,aggr=None,num_head=1,is_final_layer=True):
        super().__init__(aggr=aggr)

    def message(self,x_i,x_j,index):
        """
        x_j: source node, [num_edges,node_feat_dim]
        x_i: target node, [num_edges,node_feat_dim]
        index: target node indices, [num_edges,]
        """
    
    def aggregate(self,inputs,index):
        """
        inputs: [num_edges,num_head,latent_dim]
        index:  target node indices, [num_edges,]
        """
    
    def forward(self,x,edge_index):
        h=self.propagate(edge_index=edge_index,x=x)
        return h