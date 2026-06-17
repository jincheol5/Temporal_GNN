import torch
import torch.nn as nn
from typing_extensions import Literal
from data import TemporalGraph,Memory
from module import TimeEncoder,GraphAttnEmbedding,MemoryUpdater

class TGN_Base(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            mem_dim:int,
            latent_dim:int,
            msg_dim:int,
            time_dim:int,
            output_dim:int,
            graph:TemporalGraph,
            memory:Memory,
            n_layer:int,
            n_neighbor:int,
            n_head:int,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.mem_dim=mem_dim
        self.latent_dim=latent_dim
        self.msg_dim=msg_dim
        self.time_dim=time_dim
        self.output_dim=output_dim
        self.graph=graph
        self.memory=memory
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # memory updater
        self.memory_updater=MemoryUpdater(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            time_encoder=self.time_encoder,
            graph=self.graph,
            memory=self.memory,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            mem_dim=mem_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            graph=self.graph,
            memory=self.memory,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            use_memory=True,
            time_encoder=self.time_encoder
        )

        # pre batch data
        self.pre_batch=False
        self.pre_src=None
        self.pre_dst=None
        self.pre_edge=None
        self.pre_event_t=None
    
    def set_pre_batch(self,
            pre_src:torch.Tensor,
            pre_dst:torch.Tensor,
            pre_edge:torch.Tensor,
            pre_event_t:torch.Tensor
        ):
        self.pre_batch=True
        self.pre_src=pre_src.detach()
        self.pre_dst=pre_dst.detach()
        self.pre_edge=pre_edge.detach()
        self.pre_event_t=pre_event_t.detach()

    def forward(self):
        return NotImplemented

class TGN_Link_Prediction(TGN_Base):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            mem_dim:int,
            latent_dim:int,
            msg_dim:int,
            time_dim:int,
            output_dim:int,
            graph:TemporalGraph,
            memory:Memory,
            n_layer:int,
            n_neighbor:int,
            n_head:int,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super(TGN_Link_Prediction,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            mem_dim=mem_dim,
            latent_dim=latent_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            graph=graph,
            memory=memory,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )
        # decoder
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=output_dim+output_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=1
            )
        )

    def forward(self,
            pos_event:dict,
            neg_event:dict
        ):
        """
        Input:
            pos_event: dict
                key: src, dst, edge, event_t
                value: 
                    src: [B,] 
                    dst: [B,] 
                    edge: [B,]
                    event_t: [B,]
            neg_event: dict
                key: src, dst, edge, event_t
                value:
                    src: [B,] 
                    dst: [B,]
                    edge: [B,] 
                    event_t: [B,]
        """
        ### 0. unpack event dict
        pos_src=pos_event["src"]
        pos_dst=pos_event["dst"]
        pos_edge=pos_event["edge"]
        pos_event_t=pos_event["event_t"]
        neg_src=neg_event["src"]
        neg_dst=neg_event["dst"]
        neg_edge=neg_event["edge"]
        neg_event_t=neg_event["event_t"]

        src=torch.concat([pos_src,neg_src],dim=0) # [2B,]
        dst=torch.concat([pos_dst,neg_dst],dim=0) # [2B,]
        edge=torch.concat([pos_edge,neg_edge],dim=0) # [2B,]
        event_t=torch.concat([pos_event_t,neg_event_t],dim=0) # [2B,]

        ### 1. pre batch에 대한 memory update
        if self.pre_batch:
            self.memory_updater.update_memory(
                src=self.pre_src,
                dst=self.pre_dst,
                edge=self.pre_edge,
                event_t=self.pre_event_t
            )

        ### 2. pre batch에 current batch setting
        self.set_pre_batch(
            pre_src=pos_src,
            pre_dst=pos_dst,
            pre_edge=edge,
            pre_event_t=pos_event_t
        )

        ### 3. current batch에 대한 embedding
        batch_size=src.size(0) # 2B

        # concat
        tar=torch.concat([src,dst],dim=0) # [4B,]
        tar_t=torch.cat([event_t,event_t],dim=0) # [4B,]

        # encoding
        tar_ft=self.encoder.compute_embedding(
            tar=tar,
            tar_t=tar_t,
            n_layer=self.n_layer
        ) # [4B,output_dim]

        # split to src,dst
        src_ft=tar_ft[:batch_size]
        dst_ft=tar_ft[batch_size:]
        
        link_ft=torch.concat([src_ft,dst_ft],dim=-1) # [2B,output_dim+output_dim]
        pred_link_logit=self.decoder(link_ft) # [2B,1]
        return pred_link_logit