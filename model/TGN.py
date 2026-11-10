import torch
import torch.nn as nn
from typing_extensions import Literal
from graph import GNN_Graph
from module import TimeEncoder,Memory,GraphAttnEmbedding,GRUMemoryUpdater,MLP

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
            aggr_fn:Literal["last","mean"]="last",
            msg_fn:Literal["concat","mlp"]="concat"
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
        self.aggr_fn=aggr_fn
        self.msg_fn=msg_fn
        
        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # memory updater
        self.memory_updater=GRUMemoryUpdater(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            msg_dim=msg_dim,
            time_encoder=self.time_encoder,
            aggr_fn=aggr_fn,
            msg_fn=msg_fn
        )

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            mem_dim=mem_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            time_encoder=self.time_encoder,
            use_memory=True
        )

        # decoder
        self.decoder=MLP(
            input_dim=2*embed_dim,
            latent_dim=latent_dim,
            output_dim=1
        )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node
        self.memory=Memory(
            n_node=self.n_node,
            mem_dim=self.mem_dim,
            edge_dim=self.edge_dim,
            time_dim=self.time_dim
        )
        self.encoder.set_graph(graph=graph)
        self.memory_updater.set_graph(graph=graph)

    def update_memory(self):
        """
        model의 memory state 업데이트
        """
        # get pre_msg_store
        pre_msg_store=self.memory.get_pre_msg_store()
        node=pre_msg_store["node"]
        pre_msg=pre_msg_store["pre_msg"]
        pre_msg_t=pre_msg_store["pre_msg_t"]

        # update memory about pre_msg_store
        updated_result=self.memory_updater.update_memory(
            node=node,
            msg=pre_msg,
            msg_t=pre_msg_t,
            mem_vec=self.memory.mem_vec
        )
        self.memory.update_memory_state(
            node=updated_result["node"],
            mem_vec=updated_result["mem_vec"],
            mem_t=updated_result["mem_t"]
        )

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        batch_size=src.size(0)
        tar=torch.concat([src,dst],dim=0) 
        tar_t=torch.cat([event_t,event_t],dim=0)
        embedded_tar_vec=self.encoder.compute_embedding(
            tar=tar,
            tar_t=tar_t,
            mem_vec=self.memory.mem_vec,
            n_layer=self.n_layer
        )
        src_vec=embedded_tar_vec[:batch_size]
        dst_vec=embedded_tar_vec[batch_size:]
        return {
            "src_vec":src_vec,
            "dst_vec":dst_vec
        }

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        ### embedding
        embedded_vec=self.embedding(
            src=src,
            dst=dst,
            event_t=event_t
        )
        src_vec=embedded_vec["src_vec"]
        dst_vec=embedded_vec["dst_vec"]

        ### decode 
        pair_vec=torch.concat([src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]