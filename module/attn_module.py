import math
import torch
import torch.nn as nn

class TemporalGraphAttn(nn.Module):
    def __init__(self,
            input_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            embed_dim:int,
            n_head:int=1,
            dropout:float=0.1
        ):
        super().__init__()
        self.input_dim=input_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.embed_dim=embed_dim
        self.n_head=n_head
        self.attn_dim=input_dim+edge_dim+time_dim
        if n_head<=0 or self.attn_dim % n_head!=0:
            raise ValueError("input_dim + edge_dim + time_dim must be divisible by n_head")
        self.head_dim=self.attn_dim//n_head
        self.q_dim=self.attn_dim
        self.kv_dim=self.attn_dim

        # One bias-free projection per Q/K/V, matching official MultiHeadAttention.
        self.query_proj=nn.Linear(self.attn_dim,self.attn_dim,bias=False)
        self.key_proj=nn.Linear(self.attn_dim,self.attn_dim,bias=False)
        self.value_proj=nn.Linear(self.attn_dim,self.attn_dim,bias=False)
        for projection in (self.query_proj,self.key_proj,self.value_proj):
            nn.init.normal_(projection.weight,mean=0,
                            std=math.sqrt(2.0/(self.attn_dim+self.head_dim)))
        self.output_proj=nn.Linear(self.attn_dim,self.attn_dim)
        nn.init.xavier_normal_(self.output_proj.weight)
        self.attn_dropout=nn.Dropout(dropout)
        self.output_dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(self.attn_dim)
        self.MLPs=nn.Sequential(
            nn.Linear(self.attn_dim+input_dim,input_dim),
            nn.ReLU(),
            nn.Linear(input_dim,embed_dim)
        )
        nn.init.xavier_normal_(self.MLPs[0].weight)
        nn.init.xavier_normal_(self.MLPs[2].weight)

    def forward(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor
        ):
        batch_size,n_neighbor=neighbor_mask.shape
        # Root has no incoming edge. Preserve this unprojected query as residual.
        query=torch.cat((tar_vec,tar_vec.new_zeros(batch_size,self.edge_dim),tar_ts_vec),dim=-1).unsqueeze(1)
        key=torch.cat((neighbor_vec,neighbor_edge_ft,neighbor_ts_vec),dim=-1)
        residual=query
        q=self.query_proj(query).reshape(batch_size,1,self.n_head,self.head_dim).transpose(1,2)
        k=self.key_proj(key).reshape(batch_size,n_neighbor,self.n_head,self.head_dim).transpose(1,2)
        v=self.value_proj(key).reshape(batch_size,n_neighbor,self.n_head,self.head_dim).transpose(1,2)

        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        scores=scores.masked_fill(~neighbor_mask[:,None,None,:],-1e10)
        weights=self.attn_dropout(torch.softmax(scores,dim=-1))
        output=torch.matmul(weights,v).transpose(1,2).reshape(batch_size,1,self.attn_dim)
        output=self.output_dropout(self.output_proj(output))
        # With no history, contribute no neighbor message, but retain the query.
        output=output.masked_fill(~neighbor_mask.any(dim=-1)[:,None,None],0)
        output=self.layer_norm(output+residual).squeeze(1)
        return self.MLPs(torch.cat((output,tar_vec),dim=-1)) # [B,embed_dim]

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block

    Block 구성:
        1. Multi-head Self-Attention (MSA) part
            1-1. Layer Normalization
            1-2. Multi-head Self-Attention
            1-3. Residual connection
        2. Feed-Forward Network (FFN) part
            2-1. Layer Normalization
            2-2. Feed-Forward Network
            2-3. Residual connection
    
    FFN 내부 비활성화 함수=GELU 사용
    """
    def __init__(self,
            attn_dim:int,
            latent_dim:int,
            n_head:int,
            dropout:float=0.1
        ):
        super().__init__()
        self.multi_head_attention=nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.dropout=nn.Dropout(dropout)
        self.linear_layers=nn.ModuleList([
            nn.Linear(in_features=attn_dim,out_features=latent_dim),
            nn.Linear(in_features=latent_dim,out_features=attn_dim)
        ])
        self.norm_layers=nn.ModuleList([
            nn.LayerNorm(attn_dim),
            nn.LayerNorm(attn_dim)
        ])
        self.gelu=nn.GELU()
    
    def forward(self,z:torch.Tensor):
        ###### 1. MSA part
        ### Pre-LN
        norm_z=self.norm_layers[0](z) 

        ### Self-Attention: query = key = value = z
        attn_out,_=self.multi_head_attention(
            query=norm_z,
            key=norm_z,
            value=norm_z
        ) 

        ### Residual connection
        out=z+self.dropout(attn_out)  

        ###### 2. FFN part
        ### Pre-LN
        norm_out=self.norm_layers[1](out) 

        ### FFN
        hidden=self.linear_layers[0](norm_out) 
        hidden=self.gelu(hidden)
        hidden=self.dropout(hidden)
        hidden=self.linear_layers[1](hidden)

        ### Residual connection
        out=out+self.dropout(hidden) 
        return out