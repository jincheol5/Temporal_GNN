import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Input:
            timestamps: [batch_size,N,1]
        Output:
            updated_timestamps: [batch_size,N,time_dim]
        """
        output=torch.cos(self.w(timestamps)) # [batch_size,N,time_dim]
        return output

class Attention(nn.Module):
    def __init__(self,node_dim,latent_dim):
        super().__init__()
        self.query_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.key_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.value_linear=nn.Linear(in_features=node_dim+latent_dim+latent_dim,out_features=latent_dim)
        self.ffn=nn.Sequential(
            nn.Linear(latent_dim+node_dim+latent_dim+latent_dim,latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim,latent_dim)
        )

    def forward(self,target,z,neighbor_mask):
        """
        Input:
            target: [batch_size,node_dim+latent_dim+latent_dim], target node feature||time_feature
            z: [batch_size,N,node_dim+latent_dim+latent_dim], all node feature||time_feature
            neighbor_mask: [batch_size,N,], neighbor node mask
        """
        feature_dim=target.size(1)

        q=self.query_linear(target) # [batch_size,latent_dim]
        k=self.key_linear(z) # [batch_size,N,latent_dim]
        v=self.value_linear(z) # [batch_size,N,latent_dim]

        q=q.unsqueeze(1) # [batch_size,1,latent_dim]
        attention_scores=torch.matmul(q,k.transpose(1,2))/(feature_dim**0.5) # [batch_size,1,N]

        neighbor_mask=neighbor_mask.unsqueeze(1) # [batch_size,1,N]
        attention_scores=attention_scores.masked_fill(~neighbor_mask,float('-inf')) # [batch_size,1,N]

        attention_weight=F.softmax(attention_scores,dim=-1) # [batch_size,1,N]

        neighbor_weight_sum=torch.matmul(attention_weight,v) # [batch_size,1,latent_dim]
        neighbor_weight_sum=neighbor_weight_sum.squeeze(1) # [batch_size,latent_dim]

        h_input=torch.cat([neighbor_weight_sum,target],dim=-1) # [batch_size,latent_dim||node+latent_dim+latent_dim]
        h_output=self.ffn(h_input) # [batch_size,latent_dim]
        return h_output
