import torch
import torch.nn as nn
from data import TemporalGraph
from module import TimeEncoder,GraphAttnEmbedding

class TGAT_Base(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            output_dim:int,
            graph:TemporalGraph,
            n_layer:int,
            n_neighbor:int,
            n_head:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.output_dim=output_dim
        self.graph=graph
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            graph=self.graph,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            use_memory=False,
            time_encoder=self.time_encoder
        )

    def forward(self):
        return NotImplemented
    
class TGAT_Link_Prediction(TGAT_Base):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            output_dim:int,
            graph:TemporalGraph,
            n_layer:int,
            n_neighbor:int,
            n_head:int
        ):
        super(TGAT_Link_Prediction,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            graph=graph,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head
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
        pos_event_t=pos_event["event_t"]
        neg_src=neg_event["src"]
        neg_dst=neg_event["dst"]
        neg_event_t=neg_event["event_t"]

        src=torch.concat([pos_src,neg_src],dim=0) # [2B,]
        dst=torch.concat([pos_dst,neg_dst],dim=0) # [2B,]
        event_t=torch.concat([pos_event_t,neg_event_t],dim=0) # [2B,]

        ### 1. current batch에 대한 embedding
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