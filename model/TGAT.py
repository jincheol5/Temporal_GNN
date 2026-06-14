import torch
import torch.nn as nn
from data import TemporalGraph
from module import TimeEncoder,GraphAttnEmbedding

class TGAT_Base(nn.Module):
    def __init__(self,
            node_dim:int,
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
            src,
            dst,
            event_t,
        ):
        """
        Input:
            src: [B,]
            dst: [B,]
            event_t: [B,]
            B = pos_E + neg_E in batch
        """
        batch_size=src.size(0)

        # concat
        tar=torch.concat([src,dst],dim=0) # [2B,]
        tar_t=torch.cat([event_t,event_t],dim=0) # [2B,]

        # encoding
        tar_ft=self.encoder.compute_embedding(
            tar=tar,
            tar_t=tar_t,
            n_layer=self.n_layer
        ) # [2B,output_dim]

        # split to src,dst
        src_ft=tar_ft[:batch_size]
        dst_ft=tar_ft[batch_size:]
        
        edge_ft=torch.concat([src_ft,dst_ft],dim=-1) # [B,output_dim+output_dim]
        pred_edge_logit=self.decoder(edge_ft) # [B,1]
        return pred_edge_logit