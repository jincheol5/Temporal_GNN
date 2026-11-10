import torch
import torch.nn as nn
from .mlp_module import MLP

class TemporalGraphAttn(nn.Module):
    def __init__(self,
            input_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            embed_dim:int,
            n_head:int,
            dropout:float=0.1
        ):
        super().__init__()
        self.input_dim=input_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.embed_dim=embed_dim

        # set multi-head attention module
        self.q_dim=input_dim+time_dim
        self.kv_dim=input_dim+time_dim+edge_dim
        self.multi_head_attn=nn.MultiheadAttention(
            embed_dim=self.q_dim,
            kdim=self.kv_dim,
            vdim=self.kv_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        # set MLP
        self.mlp=MLP(
            input_dim=self.q_dim+input_dim,
            latent_dim=latent_dim,
            output_dim=embed_dim
        )

    def forward(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor
        ):
        """
        Input:
            tar_vec: [n_tar,input_dim]
            tar_ts_vec: [n_tar,time_dim]
            neighbor_vec: [n_tar,n_neighbor,input_dim]
            neighbor_ts_vec: [n_tar,n_neighbor,time_dim]
            neighbor_edge_ft: [n_tar,n_neighbor,edge_dim]
            neighbor_mask: [n_tar,n_neighbor]
        """
        tar_vec=tar_vec.unsqueeze(dim=1) # [n_tar,input_dim] -> [n_tar,1,input_dim]
        tar_ts_vec=tar_ts_vec.unsqueeze(dim=1) # [n_tar,input_dim] -> [n_tar,1,input_dim]

        query=torch.cat([tar_vec,tar_ts_vec],dim=2) # [n_tar,1,input_dim+time_dim]
        key=torch.cat([neighbor_vec,neighbor_ts_vec,neighbor_edge_ft],dim=2) # [n_tar,1,input_dim+time_dim+edge_dim]
        value=torch.cat([neighbor_vec,neighbor_ts_vec,neighbor_edge_ft],dim=2) # [n_tar,1,input_dim+time_dim+edge_dim]

        ### 유효한 neighbor가 없는 tar 찾은 후 전처리
        # neighbor가 없는 tar의 경우 Attention 계산이 끝난 후 해당 결과를 다시 0으로 변경.
        neighbor_mask=~neighbor_mask # key_padding_mask로 처리하기 위해 반전 연산. nn.MultiheadAttention은 key_padding_mask=True를 제외.
        invalid_neighbor_mask=neighbor_mask.all(dim=1) # [n_tar,]
        neighbor_mask[invalid_neighbor_mask,0]=False # 유효한 neighbor가 없는 tar의 첫 번째를 임시로 False(이웃 존재) 처리.

        attn_output,_=self.multi_head_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=neighbor_mask, # true를 제외
            need_weights=False
        ) # [n_tar,1,q_dim]
        attn_output=attn_output.squeeze(1) # [n_tar,q_dim]

        # 유효한 Neighbor가 없는 Target의 Attention Output 제거
        attn_output=attn_output.masked_fill(invalid_neighbor_mask.unsqueeze(1),0.0)

        # Target Feature + Neighbor Attention Output
        tar_vec=tar_vec.squeeze(dim=1)
        tar_vec=torch.cat([attn_output,tar_vec],dim=-1) # [n_tar,q_dim+input_dim]
        output=self.mlp(tar_vec) # [n_tar,embed_dim]
        return output 