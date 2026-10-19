import torch
import torch.nn as nn
from typing_extensions import Literal
from graph import GNN_Graph
from module import TimeEncoder,Memory,GraphAttnEmbedding,GRUMemoryUpdater,Link_Predictor

class TGN(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            msg_dim:int,
            mem_dim:int,
            embed_dim:int,
            n_layer:int,
            n_neighbor:int,
            n_head:int,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.msg_dim=msg_dim
        self.mem_dim=mem_dim
        self.embed_dim=embed_dim

        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # memory updater
        self.memory_updater=GRUMemoryUpdater(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            time_encoder=self.time_encoder,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            latent_dim=latent_dim,
            mem_dim=mem_dim,
            embed_dim=embed_dim,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            use_memory=True,
            time_encoder=self.time_encoder
        )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node
        self.memory=Memory(n_node=self.n_node,mem_dim=self.mem_dim)
        self.encoder.set_graph(graph=graph)
        self.memory_updater.set_graph(graph=graph)

    def update_memory(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            edge:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        eventstream에 대해서 model의 memory state 업데이트
        """
        mem_vec=self.memory.get_mem_vec()
        mem_t=self.memory.get_mem_t()
        updated_result=self.memory_updater.update_memory(
            src=src,
            dst=dst,
            edge=edge,
            event_t=event_t,
            mem_vec=mem_vec,
            mem_t=mem_t
        )
        updated_node=updated_result["node"]
        updated_mem_vec=updated_result["mem_vec"]
        updated_mem_t=updated_result["mem_t"]
        self.memory.update_memory_state(
            node=updated_node,
            mem_vec=updated_mem_vec,
            mem_t=updated_mem_t
        )

    def embedding(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor
        ):
        mem_vec=self.memory.get_mem_vec()
        embedded_tar_vec=self.encoder.compute_embedding(
            tar=tar,
            tar_t=tar_t,
            mem_vec=mem_vec,
            n_layer=self.n_layer
        )
        return embedded_tar_vec

class TGN_LP(TGN):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            msg_dim:int,
            mem_dim:int,
            embed_dim:int,
            n_layer:int,
            n_neighbor:int,
            n_head:int,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super(TGN_LP,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            latent_dim=latent_dim,
            msg_dim=msg_dim,
            mem_dim=mem_dim,
            embed_dim=embed_dim,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )

        # decoder
        self.decoder=Link_Predictor(
            embed_dim=embed_dim,
            latent_dim=latent_dim
        )
