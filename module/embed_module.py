import math
import torch
import torch.nn as nn
from data import TemporalGraph,Memory
from .time_encoder import TimeEncoder
from .aggr_module import TemporalGraphAttn

class EmbeddingModule(nn.Module):
    def __init__(self,
            node_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32
        ):
        super().__init__()
        self.node_dim=node_dim
        self.latent_dim=latent_dim
        self.output_dim=output_dim

    def compute_embedding(self):
        return NotImplemented

class IdentityEmbedding(EmbeddingModule):
    """
    memory state를 node embedding으로 직접 사용
    """
    def __init__(self,
            node_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            memory:Memory=None,
        ):
        super(IdentityEmbedding,self).__init__(
            node_dim=node_dim,
            latent_dim=latent_dim,
            output_dim=output_dim
        )
        self.memory=memory
        self.mem_dim=mem_dim
    def compute_embedding(self,tar):
        tar_ft=self.memory.get_memory(node=tar) # [B,mem_dim]
        return tar_ft

class TimeProjectionEmbedding(EmbeddingModule):
    """
    emb(i,t)=(1+delta_t*w) x s^t_i
    """
    def __init__(self,
            node_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32,
            output_dim:int=32,
            memory:Memory=None,
        ):
        super(TimeProjectionEmbedding,self).__init__(
            node_dim=node_dim,
            latent_dim=latent_dim,
            output_dim=output_dim
        )
        self.memory=memory
        self.mem_dim=mem_dim
        # time-projection embedding layer
        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv=1./math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0,stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0,stdv)
        self.embedding_layer=NormalLinear(in_features=1,out_features=self.mem_dim)

    def compute_embedding(self,tar,tar_t):
        tar_mem=self.memory.get_memory(node=tar) # [B,mem_dim]
        tar_ts=self.memory.get_node_timespan(
            node=tar,
            event_t=tar_t
        ) # [B,1]
        tar_ft=tar_mem*(1+self.embedding_layer(tar_ts)) # [B,mem_dim]
        return tar_ft

