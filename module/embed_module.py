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


class GraphEmbeddingModule(EmbeddingModule):
    def __init__(self,
            node_dim:int=32,
            mem_dim:int=32,
            latent_dim:int=32, 
            output_dim:int=32,
            time_dim:int=32,
            graph:TemporalGraph=None,
            memory:Memory=None,
            n_layer:int=1,
            use_memory:bool=True,
            time_encoder:TimeEncoder=None
        ):
        super(GraphEmbeddingModule,self).__init__(
            node_dim,
            latent_dim,
            output_dim
        )
        self.time_dim=time_dim
        self.graph=graph
        self.n_layer=n_layer
        self.use_memory=use_memory
        if use_memory:
            self.mem_dim=mem_dim
            self.memory=memory

        # module
        self.time_encoder=time_encoder

    def aggregate(self,
            tar_ft:torch.Tensor,
            tar_ts_ft:torch.Tensor,
            neighbor_ft:torch.Tensor,
            neighbor_ts_ft:torch.Tensor,
            neighbor_mask:torch.Tensor,
            n_layer:int
        ):
        return NotImplemented

    def compute_embedding(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            n_layer:int=1,
            n_neighbor:int=10
        ):
        """
        embedding 할 노드들
        embedding 할 노드들의 이웃 노드들

        Input:
            tar: [n_tar,]
            tar_t: [n_tar,]
            n_layer: int
            n_neighbor: int
        Return:
        """
        tar_ft=self.graph.get_node_ft(node=tar) # [n_tar,node_dim]
        if self.use_memory:
            tar_mem=self.memory.get_memory(node=tar)
            tar_ft=torch.concat(
                [tar_mem,tar_ft],
                dim=-1
            ) # [n_tar,node_dim+mem_dim]

        if n_layer==0:
            return tar_ft
        else:
            """
            neighbor_id: [n_tar,n_neighbor]
            neighbor_t: [n_tar,n_neighbor]
            """
            neighbor_id,neighbor_t,neighbor_ts,_=self.graph.get_temporal_neighbor(
                tar=tar,
                tar_t=tar_t,
                n_neighbor=n_neighbor
            )

            # flatten for neighbor embedding
            n_tar,n_neighbor=neighbor_id.size()
            neighbor_id=neighbor_id.flatten() # [n_tar,n_neighbor] -> [n_tar x n_neighbor,]
            neighbor_t=neighbor_t.flatten() # [n_tar,n_neighbor] -> [n_tar x n_neighbor,]

            # compute neighbor_mask
            neighbor_mask=neighbor_id!=0 # [n_tar x n_neighbor,], bool
            
            # apply time encoding
            tar_ts=torch.zeros_like(tar,device=tar.device).unsqueeze(-1) # [n_tar,1]
            tar_ts_ft=self.time_encoder(tar_ts) # [n_tar,time_dim]
            neighbor_ts=neighbor_ts.unsqueeze(-1) # [n_tar,n_neighbor,1]
            neighbor_ts_ft=self.time_encoder(neighbor_ts) # [n_tar,n_neighbor,time_dim]

            # get neighbor embedding
            neighbor_ft=self.compute_embedding(
                tar=neighbor_id,
                tar_t=neighbor_t,
                n_layer=n_layer-1,
                n_neighbor=n_neighbor
            ) # [n_tar x n_neighbor,output_dim]

            # reshape
            neighbor_ft=neighbor_ft.reshape(n_tar,n_neighbor,-1) # -> [n_tar,n_neighbor,output_dim]
            neighbor_mask=neighbor_mask.reshape(n_tar,n_neighbor,) # -> [n_tar,n_neighbor]

            ### Aggregation
            updated_tar_ft=self.aggregate(
                tar_ft=tar_ft,
                tar_ts_ft=tar_ts_ft,
                neighbor_ft=neighbor_ft,
                neighbor_ts_ft=neighbor_ts_ft,
                neighbor_mask=neighbor_mask,
                n_layer=n_layer
            )
            return updated_tar_ft



