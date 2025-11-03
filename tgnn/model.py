import torch
import torch.nn as nn
from .modules import TimeEncoder,MemoryUpdater,Attention

class TGAT(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.attention=Attention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim

    def forward(self,target,x,t,neighbor_mask):
        """
        Input:
            target: [batch_size,1], long
            x: [batch_size,N,1], float
            t: [batch_size,N,1], float
            neighbor_mask: [batch_size,N,], neighbor node mask
        Output:
            logit: [batch_size,1]
        """
        batch_size,num_nodes,_=x.size()

        # target vector
        batch_indices=torch.arange(batch_size,device=x.device) # [batch_size,]
        target=target.squeeze(-1) # [batch_size,]
        target_vec=x[batch_indices,target,:] # [batch_size,1]

        # target
        target_hidden_ft=torch.zeros((batch_size,self.latent_dim),dtype=torch.float,device=x.device) # [batch_size,latent_dim]
        target_vec=torch.cat([target_vec,target_hidden_ft],dim=-1) # [batch_size,node_dim+latent_dim]
        target_t=torch.zeros((batch_size,1),dtype=torch.float,device=x.device) # [batch_size,1]
        encoded_target_t=self.time_encoder(target_t) # [batch_size,latent_dim]
        target_h=torch.cat([target_vec,encoded_target_t],dim=-1) # [batch_size,node_dim+latent_dim+latent_dim]

        # neighbor
        hidden_ft=torch.zeros((batch_size,num_nodes,self.latent_dim),dtype=torch.float,device=x.device) # [batch_size,N,latent_dim]
        x=torch.cat([x,hidden_ft],dim=-1) # [batch_size,N,node_dim+latent_dim]
        encoded_t=self.time_encoder(t) # [batch_size,N,latent_dim]
        h=torch.cat([x,encoded_t],dim=-1) # [batch_size,N,node_dim+latent_dim+latent_dim]

        # attention result
        z=self.attention(target=target_h,h=h,neighbor_mask=neighbor_mask) # [batch_size,latent_dim]
        logit=self.linear(z) # [batch_size,1]
        return logit

class TGN(nn.Module):
    def __init__(self,node_dim,latent_dim): 
        super().__init__()
        self.time_encoder=TimeEncoder(time_dim=latent_dim)
        self.memory_updater=MemoryUpdater(node_dim=node_dim,latent_dim=latent_dim)
        self.attention=Attention(node_dim=node_dim,latent_dim=latent_dim)
        self.linear=nn.Linear(in_features=latent_dim,out_features=1)
        self.latent_dim=latent_dim

    def forward(self,batch_list):
        """
        Input:
            List of batch_dict:
                source: [batch_size,1], long
                target: [batch_size,1], long
                x: [batch_size,N,1], float
                memory: [N,latent_dim], float 
                t: [batch_size,N,1], float
                neighbor_mask: [batch_size,N,], neighbor node mask
        Output:
            logit: [seq_len,batch_size,1]
        """
        logit_list=[]
        num_nodes=batch_list[0]['x'].size(1)
        device=batch_list[0]['x'].device
        memory=torch.zeros(num_nodes,self.latent_dim,dtype=torch.float32,device=batch_list[0]['x'].device)
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
            z=self.attention(target=target_h,h=h,neighbor_mask=neighbor_mask) # [batch_size,latent_dim]
            logit=self.linear(z) # [batch_size,1]

            logit_list.append(logit)
            memory=updated_memory # set next memory

        output=torch.stack(logit_list,dim=0) # [seq_len,batch_size,1]
        return output # [seq_len,batch_size,1]