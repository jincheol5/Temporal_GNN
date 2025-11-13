import torch
import torch.nn as nn
from typing_extensions import Literal
from .graph_utils import GraphUtils
from .modules import TimeEncoder,MemoryUpdater,TimeProjection,GraphAttention,TemporalGraphSum

class TGAT(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim

    def forward(self,data_loader,device):
        """
        Input:
            data_loader: List of batch_event_dict
                batch_event_dict
                    x: [B,N,1]
                    t: [B,N,1]
                    tar: [B,1]
                    n_mask: [B,N,]
                    edge_index: [2,E]
            device: GPU
        Output:
            step_logit: [seq_len,batch_size,1]
            last_logit: [N,1]
        """
        """
        compute step tR logit
        """
        logit_list=[]
        for batch in data_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            x=batch['x'] # [B,N,1], float
            t=batch['t'] # [B,N,1], float
            tar=batch['tar'] # [B,1], long
            n_mask=batch['n_mask'] # [B,N,], neighbor node mask

            """
            embedding
            """
            batch_size,num_nodes,_=x.size()

            # target x vector
            batch_idx=torch.arange(batch_size,device=x.device) # [batch_size,]
            tar=tar.squeeze(-1) # [batch_size,]
            tar_x=x[batch_idx,tar,:] # [batch_size,1]

            # target vector
            tar_hidden_ft=torch.zeros((batch_size,self.latent_dim),dtype=torch.float,device=x.device) # [batch_size,latent_dim]
            tar_vec=torch.cat([tar_x,tar_hidden_ft],dim=-1) # [batch_size,node_dim+latent_dim]
            tar_t=torch.zeros((batch_size,1),dtype=torch.float,device=x.device) # [batch_size,1]
            encoded_tar_t=self.time_encoder(tar_t) # [batch_size,latent_dim]
            tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [batch_size,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=torch.zeros((batch_size,num_nodes,self.latent_dim),dtype=torch.float,device=x.device) # [batch_size,N,latent_dim]
            x=torch.cat([x,hidden_ft],dim=-1) # [batch_size,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(t) # [batch_size,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [batch_size,N,node_dim+latent_dim+latent_dim]

            # attention result
            z=self.attention(target=tar_h,h=h,neighbor_mask=n_mask) # [batch_size,latent_dim]
            logit=self.linear(z) # [batch_size,1]
            logit_list.append(logit)
        step_logit=torch.stack(logit_list,dim=0) # [seq_len,batch_size,1]

        """
        compute last tR logit
        """
        x=data_loader[-1]['x'][0] # [N,1]
        t=data_loader[-1]['t'][0] # [N,1]
        edge_index=data_loader[-1]['edge_index'][0] # [2,E]
        adj_mask=GraphUtils.get_adj_mask(num_nodes=num_nodes,edge_index=edge_index) # [N,N]

        # target vector (all nodes)
        tar_hidden_ft=torch.zeros((num_nodes,self.latent_dim),dtype=torch.float,device=x.device) # [N,latent_dim]
        tar_vec=torch.cat([x,tar_hidden_ft],dim=-1) # [N,node_dim+latent_dim]
        tar_t=torch.zeros((num_nodes,1),dtype=torch.float,device=x.device) # [N,1]
        encoded_tar_t=self.time_encoder(tar_t) # [N,latent_dim]
        tar_h=torch.cat([tar_vec,encoded_tar_t],dim=-1) # [N,node_dim+latent_dim+latent_dim]

        # neighbor
        hidden_ft=torch.zeros((num_nodes,self.latent_dim),dtype=torch.float,device=x.device) # [N,latent_dim]
        x=torch.cat([x,hidden_ft],dim=-1) # [N,latent_dim]
        encoded_t=self.time_encoder(t) # [N,latent_dim]
        h=torch.cat([x,encoded_t],dim=-1) # [N,node_dim+latent_dim+latent_dim]

        # attention result
        z=self.attention(target=tar_h,h=h,neighbor_mask=adj_mask,step='last') # [N,latent_dim]
        last_logit=self.linear(z) # [N,1]

        output={}
        output['step_logit']=step_logit # [seq_len,batch_size,1]
        output['last_logit']=last_logit # [N,1]
        return output 

class TGN(nn.Module):
    def __init__(self,node_dim,latent_dim,emb:Literal['time','attn','sum']): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(node_dim=node_dim,latent_dim=latent_dim)
        match emb:
            case 'time':
                self.embedding=TimeProjection(latent_dim=latent_dim)
            case 'attn':
                self.embedding=GraphAttention(node_dim=node_dim,latent_dim=latent_dim)
            case 'sum':
                self.embedding=TemporalGraphSum(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim
        self.emb=emb

    def forward(self,batch_list,device):
        """
        Input:
            List of batch_dict:
                source: [batch_size,1], long
                target: [batch_size,1], long
                x: [batch_size,N,1], float
                memory: [N,latent_dim], float 
                t: [batch_size,N,1], float
                neighbor_mask: [batch_size,N,], neighbor node mask
            device: GPU
        Output:
            logit: [seq_len,batch_size,1]
        """
        logit_list=[]
        num_nodes=batch_list[0]['x'].size(1)
        memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=device)
        for batch in batch_list:
            batch={k:v.to(device) for k,v in batch.items()}
            source=batch['source'] # [batch_size,1], long
            target=batch['target'] # [batch_size,1], long
            x=batch['x'] # [batch_size,N,1], float
            t=batch['t'] # [batch_size,N,1], float
            neighbor_mask=batch['neighbor_mask'] # [batch_size,N,], neighbor node mask

            """
            1. update memory
            """
            delta_t=self.time_encoder(t) # [batch_size,N,latent_dim]
            updated_memory=self.memory_updater(x=x,memory=memory,source=source,target=target,delta_t=delta_t) # [N,latent_dim]

            """
            2. embedding
            """
            batch_size,_,_=x.size()
        
            # target vector
            batch_indices=torch.arange(batch_size,device=x.device) # [batch_size,]
            target=target.squeeze(-1) # [batch_size,]
            target_vec=x[batch_indices,target,:] # [batch_size,1]

            # target
            target_hidden_ft=updated_memory[target] # [batch_size,latent_dim]
            target_vec=torch.cat([target_vec,target_hidden_ft],dim=-1) # [batch_size,node_dim+latent_dim]
            target_t=torch.zeros((batch_size,1),dtype=torch.float,device=x.device) # [batch_size,1]
            encoded_target_t=self.time_encoder(target_t) # [batch_size,latent_dim]
            target_h=torch.cat([target_vec,encoded_target_t],dim=-1) # [batch_size,node_dim+latent_dim+latent_dim]

            # neighbor
            hidden_ft=updated_memory.unsqueeze(0).expand(batch_size,-1,-1) # [batch_size,N,latent_dim]
            x=torch.cat([x,hidden_ft],dim=-1) # [batch_size,N,node_dim+latent_dim]
            encoded_t=self.time_encoder(t) # [batch_size,N,latent_dim]
            h=torch.cat([x,encoded_t],dim=-1) # [batch_size,N,node_dim+latent_dim+latent_dim]

            # attention result
            match self.emb:
                case 'time':
                    target_memory=updated_memory[target] # [batch_size,latent_dim]
                    delta_t=t[batch_indices,target,:] # [batch_size,1]
                    z=self.embedding(target_memory=target_memory,delta_t=delta_t) # [batch_size,latent_dim]
                case 'attn'|'sum':
                    z=self.embedding(target=target_h,h=h,neighbor_mask=neighbor_mask) # [batch_size,latent_dim]

            logit=self.linear(z) # [batch_size,1]

            logit_list.append(logit)
            memory=updated_memory # set next memory

        output=torch.stack(logit_list,dim=0) # [seq_len,batch_size,1]
        return output # [seq_len,batch_size,1]