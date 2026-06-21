import torch
import torch.nn as nn
from data import TemporalGraph
from utils import TrainUtils
from module import TimeEncoder,TransformerEncoderBlock

class DyGFormer_Base(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            output_dim:int,
            co_dim:int,
            patch_dim:int,
            patch_size:int,
            graph:TemporalGraph,
            n_layer:int,
            max_n_neighbor:int,
            n_head:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.output_dim=output_dim
        self.co_dim=co_dim
        self.patch_dim=patch_dim
        self.patch_size=patch_size
        self.graph=graph
        self.n_layer=n_layer
        self.max_n_neighbor=max_n_neighbor
        self.n_head=n_head

        # set attn_dim
        self.attn_dim=4*patch_dim

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # Neighbor Co-occurrence Encoding
        self.NCoE=nn.Sequential(
            nn.Linear(in_features=1,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=co_dim)
        )

        # patch_encoder
        self.patch_encoder=nn.ModuleDict(
            {
                "node":nn.Linear(in_features=patch_size*node_dim,out_features=patch_dim),
                "edge":nn.Linear(in_features=patch_size*edge_dim,out_features=patch_dim),
                "time":nn.Linear(in_features=patch_size*time_dim,out_features=patch_dim),
                "co":nn.Linear(in_features=patch_size*co_dim,out_features=patch_dim)
            }
        )

        # transformer encoder
        self.transformer_encoders=nn.ModuleList(
            [
                TransformerEncoderBlock(
                    attn_dim=self.attn_dim,
                    latent_dim=latent_dim,
                    n_head=n_head
                )
                for _ in range(n_layer)
            ]
        )

        # Time-aware Node Representation
        self.output_layer=nn.Linear(in_features=self.attn_dim,out_features=output_dim)

    def forward(self):
        return NotImplemented

class DyGFormer_Link_Prediction(DyGFormer_Base):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            output_dim:int,
            co_dim:int,
            patch_dim:int,
            patch_size:int,
            graph:TemporalGraph,
            n_layer:int,
            max_n_neighbor:int,
            n_head:int
        ):
        super(DyGFormer_Link_Prediction,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            output_dim=output_dim,
            co_dim=co_dim,
            patch_dim=patch_dim,
            patch_size=patch_size,
            graph=graph,
            n_layer=n_layer,
            max_n_neighbor=max_n_neighbor,
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
        device=pos_src.device

        ### 1. get sequence data
        src_seq=self.graph.get_historical_seq(node=src,event_t=event_t,max_n_neighbor=self.max_n_neighbor)
        src_seq_node=src_seq["seq_node"]
        src_seq_edge=src_seq["seq_edge"]
        src_seq_ts=src_seq["seq_ts"]

        dst_seq=self.graph.get_historical_seq(node=dst,event_t=event_t,max_n_neighbor=self.max_n_neighbor)
        dst_seq_node=dst_seq["seq_node"]
        dst_seq_edge=dst_seq["seq_edge"]
        dst_seq_ts=dst_seq["seq_ts"]

        co=self.graph.get_co_occurrence(
            src_seq_node=src_seq_node,
            dst_seq_node=dst_seq_node
        )
        src_seq_co=co["src_seq_co"]
        dst_seq_co=co["dst_seq_co"]

        ### 2. get padded sequence tensor
        seq_tensor=TrainUtils.padding_seq(
            src_seq_node=src_seq_node,
            src_seq_edge=src_seq_edge,
            src_seq_ts=src_seq_ts,
            src_seq_co=src_seq_co,
            dst_seq_node=dst_seq_node,
            dst_seq_edge=dst_seq_edge,
            dst_seq_ts=dst_seq_ts,
            dst_seq_co=dst_seq_co,
            patch_size=self.patch_size,
            device=device
        )

        src_seq_node=seq_tensor["src_seq_node"] # [2B,max_seq_len]
        src_seq_edge=seq_tensor["src_seq_edge"] # [2B,max_seq_len]
        src_seq_ts=seq_tensor["src_seq_ts"] # [2B,max_seq_len,1]
        src_seq_co=seq_tensor["src_seq_co"] # [2B,max_seq_len,2]

        dst_seq_node=seq_tensor["dst_seq_node"] # [2B,max_seq_len]
        dst_seq_edge=seq_tensor["dst_seq_edge"] # [2B,max_seq_len]
        dst_seq_ts=seq_tensor["dst_seq_ts"] # [2B,max_seq_len,1]
        dst_seq_co=seq_tensor["dst_seq_co"] # [2B,max_seq_len,2]

        ### 3. get node, edge feature tensor
        batch_size,max_seq_len=src_seq_node.size()
        src_seq_node=src_seq_node.flatten() # [2B x max_seq_len]
        src_seq_edge=src_seq_edge.flatten() # [2B x max_seq_len]
        dst_seq_node=dst_seq_node.flatten() # [2B x max_seq_len]
        dst_seq_edge=dst_seq_edge.flatten() # [2B x max_seq_len]

        src_seq_node_ft=self.graph.get_node_ft(node=src_seq_node) # [2B x max_seq_len,node_dim]
        src_seq_edge_ft=self.graph.get_edge_ft(edge=src_seq_edge) # [2B x max_seq_len,edge_dim]
        dst_seq_node_ft=self.graph.get_node_ft(node=dst_seq_node) # [2B x max_seq_len,node_dim]
        dst_seq_edge_ft=self.graph.get_edge_ft(edge=dst_seq_edge) # [2B x max_seq_len,edge_dim]

        src_seq_node_ft=src_seq_node_ft.reshape(batch_size,max_seq_len,self.node_dim) # [2B,max_seq_len,node_dim]
        src_seq_edge_ft=src_seq_edge_ft.reshape(batch_size,max_seq_len,self.edge_dim) # [2B,max_seq_len,edge_dim]
        dst_seq_node_ft=dst_seq_node_ft.reshape(batch_size,max_seq_len,self.node_dim) # [2B,max_seq_len,node_dim]
        dst_seq_edge_ft=dst_seq_edge_ft.reshape(batch_size,max_seq_len,self.edge_dim) # [2B,max_seq_len,edge_dim]

        ### 4. apply time_encoder
        src_seq_ts_ft=self.time_encoder(src_seq_ts) # [2B,max_seq_len,time_dim]
        dst_seq_ts_ft=self.time_encoder(dst_seq_ts) # [2B,max_seq_len,time_dim]

        ### 5. apply NCoE
        src_seq_co_0=src_seq_co[:,:,0:1] # [2B,max_seq_len,1]
        src_seq_co_1=src_seq_co[:,:,1:2] # [2B,max_seq_len,1]
        dst_seq_co_0=dst_seq_co[:,:,0:1] # [2B,max_seq_len,1]
        dst_seq_co_1=dst_seq_co[:,:,1:2] # [2B,max_seq_len,1]

        seq_co=torch.concat(
            [
                src_seq_co_0,
                src_seq_co_1,
                dst_seq_co_0,
                dst_seq_co_1
            ],
            dim=0
        ) # [8B,max_seq_len,1]

        seq_co_ft=self.NCoE(seq_co) # [8B,max_seq_len,co_dim]
        src_seq_co_0_ft,src_seq_co_1_ft,dst_seq_co_0_ft,dst_seq_co_1_ft=torch.chunk(
            seq_co_ft,
            chunks=4,
            dim=0
        ) # [2B,max_seq_len,co_dim]
        src_seq_co_ft=src_seq_co_0_ft+src_seq_co_1_ft # [2B,max_seq_len,co_dim]
        dst_seq_co_ft=dst_seq_co_0_ft+dst_seq_co_1_ft # [2B,max_seq_len,co_dim]

        ### 6. get patching sequence
        src_M=TrainUtils.patching_seq(
            seq_node_ft=src_seq_node_ft,
            seq_edge_ft=src_seq_edge_ft,
            seq_ts_ft=src_seq_ts_ft,
            seq_co_ft=src_seq_co_ft,
            patch_size=self.patch_size
        )
        src_M_n=src_M["M_n"] # [2B,l,node_dim x p]
        src_M_e=src_M["M_e"] # [2B,l,edge_dim x p]
        src_M_t=src_M["M_t"] # [2B,l,time_dim x p]
        src_M_c=src_M["M_c"] # [2B,l,co_dim x p]

        dst_M=TrainUtils.patching_seq(
            seq_node_ft=dst_seq_node_ft,
            seq_edge_ft=dst_seq_edge_ft,
            seq_ts_ft=dst_seq_ts_ft,
            seq_co_ft=dst_seq_co_ft,
            patch_size=self.patch_size
        )
        dst_M_n=dst_M["M_n"] # [2B,l,node_dim x p]
        dst_M_e=dst_M["M_e"] # [2B,l,edge_dim x p]
        dst_M_t=dst_M["M_t"] # [2B,l,time_dim x p]
        dst_M_c=dst_M["M_c"] # [2B,l,co_dim x p]

        ### 7. apply patch encoder and get z
        M_n=torch.concat(
            [
                src_M_n,
                dst_M_n
            ],
            dim=0
        ) # [4B,l,node_dim x p]
        M_e=torch.concat(
            [
                src_M_e,
                dst_M_e
            ],
            dim=0
        ) # [4B,l,edge_dim x p]
        M_t=torch.concat(
            [
                src_M_t,
                dst_M_t
            ],
            dim=0
        ) # [4B,l,time_dim x p]
        M_c=torch.concat(
            [
                src_M_c,
                dst_M_c
            ],
            dim=0
        ) # [4B,l,co_dim x p]

        M_n=self.patch_encoder["node"](M_n) # [4B,l,patch_dim]
        M_e=self.patch_encoder["edge"](M_e) # [4B,l,patch_dim]
        M_t=self.patch_encoder["time"](M_t) # [4B,l,patch_dim]
        M_c=self.patch_encoder["co"](M_c) # [4B,l,patch_dim]

        src_M_n,dst_M_n=torch.chunk(
            M_n,
            chunks=2,
            dim=0
        ) # [2B,l,patch_dim]
        src_M_e,dst_M_e=torch.chunk(
            M_e,
            chunks=2,
            dim=0
        ) # [2B,l,patch_dim]
        src_M_t,dst_M_t=torch.chunk(
            M_t,
            chunks=2,
            dim=0
        ) # [2B,l,patch_dim]
        src_M_c,dst_M_c=torch.chunk(
            M_c,
            chunks=2,
            dim=0
        ) # [2B,l,patch_dim]

        src_z=torch.concat(
            [
                src_M_n,
                src_M_e,
                src_M_t,
                src_M_c
            ],
            dim=-1
        ) # [2B,l,4 x patch_dim]
        dst_z=torch.concat(
            [
                dst_M_n,
                dst_M_e,
                dst_M_t,
                dst_M_c
            ],
            dim=-1
        ) # [2B,l,4 x patch_dim]
        z=torch.concat(
            [src_z,dst_z],
            dim=1
        ) # [2B,2l,4 x patch_dim]

        ### 8. apply transformer encoder
        for transformer_encoder in self.transformer_encoders:
            z=transformer_encoder(z)
        src_z,dst_z=torch.chunk(
            z,
            chunks=2,
            dim=1
        ) # [2B,l,4 x patch_dim]

        ### 9. Time-aware Node Representation
        src_h=src_z.mean(dim=1) # [2B,4 x patch_dim]
        dst_h=dst_z.mean(dim=1) # [2B,4 x patch_dim]

        h=torch.concat(
            [src_h,dst_h],
            dim=0
        ) # [4B,4 x patch_dim]
        h=self.output_layer(h) # [4B,output_dim]
        src_h,dst_h=torch.chunk(
            h,
            chunks=2,
            dim=0
        )  # [2B,output_dim]

        ### 10. compute link logit
        link_ft=torch.concat([src_h,dst_h],dim=-1) # [2B,output_dim+output_dim]
        pred_link_logit=self.decoder(link_ft) # [2B,1]
        return pred_link_logit