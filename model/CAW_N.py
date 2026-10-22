import torch
import torch.nn as nn
from graph import CAW_N_Graph
from module import MLP,TimeEncoder

class CAW_N(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            embed_dim:int,
            n_head:int,
            n_walk:int,
            walk_len:int,
            alpha:float=0.0
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.embed_dim=embed_dim
        self.n_head=n_head
        self.n_walk=n_walk
        self.walk_len=walk_len
        self.alpha=alpha

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # anonymizated_vec encoder
        self.anonymizated_vec_encoder=MLP(
            input_dim=walk_len+1,
            latent_dim=latent_dim,
            output_dim=latent_dim
        )

        # walk sequence encoder
        self.rnn_encoder=nn.LSTM(
            input_size=latent_dim+time_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True
        )

        # attn module for aggregation
        self.walk_attn=nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=4*embed_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )

    def set_graph(self,graph:CAW_N_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        device=src.device

        """
        1. Convert batch_node_caws to tensor
            anonymizated_vec_set -> [B,n_walk,walk_len+1,2,walk_len+1]
            time -> [B,n_walk,walk_len+1,1]
            time span -> [B,n_walk,walk_len+1,1], the root step has zero time span.
            mask -> [B,n_walk,walk_len+1], True for valid steps, False for padding.
        """
        batch_src_caws,batch_dst_caws=self.graph.set_based_anonymization(
            src=src,
            dst=dst,
            event_t=event_t
        )
        shape=(src.size(0),self.n_walk,self.walk_len+1)

        ### src
        batch_src_caws_vec=torch.stack(
            [vec for walks in batch_src_caws for walk in walks for vec,_ in walk]
        ).to(device=device).reshape(*shape,2,self.walk_len+1)
        batch_src_caws_t=torch.tensor(
            [time for walks in batch_src_caws for walk in walks for _,time in walk],
            dtype=event_t.dtype,device=device
        ).reshape(*shape,1)
        batch_src_caws_ts=torch.zeros_like(batch_src_caws_t)
        batch_src_caws_ts[:,:,1:,:]=batch_src_caws_t[:,:,:-1,:]-batch_src_caws_t[:,:,1:,:] 
        batch_src_caws_mask=batch_src_caws_vec.ne(0).any(dim=(-2,-1))

        ### dst
        batch_dst_caws_vec=torch.stack(
            [vec for walks in batch_dst_caws for walk in walks for vec,_ in walk]
        ).to(device=device).reshape(*shape,2,self.walk_len+1)
        batch_dst_caws_t=torch.tensor(
            [time for walks in batch_dst_caws for walk in walks for _,time in walk],
            dtype=event_t.dtype,device=device
        ).reshape(*shape,1)
        batch_dst_caws_ts=torch.zeros_like(batch_dst_caws_t)
        batch_dst_caws_ts[:,:,1:,:]=batch_dst_caws_t[:,:,:-1,:]-batch_dst_caws_t[:,:,1:,:]
        batch_dst_caws_mask=batch_dst_caws_vec.ne(0).any(dim=(-2,-1))

        """
        2. Encode each anonymizated_vec_set tensor and timespan
            encode anonymizated_vec_set
                MLP: [...,2,walk_len+1] -> [...,2,latent_dim]
                sum: [...,2,latent_dim] -> [B,n_walk,walk_len+1,latent_dim]
            encode time span
                [B,n_walk,walk_len+1,1] -> [B,n_walk,walk_len+1,time_dim]
        """
        ### src
        batch_src_caws_vec=self.anonymizated_vec_encoder(batch_src_caws_vec).sum(dim=-2)
        batch_src_caws_ts_vec=self.time_encoder(batch_src_caws_ts)

        ### dst
        batch_dst_caws_vec=self.anonymizated_vec_encoder(batch_dst_caws_vec).sum(dim=-2)
        batch_dst_caws_ts_vec=self.time_encoder(batch_dst_caws_ts)

        """
        3. Encode walk using RNN
        """
        batch_src_caws_vec=torch.cat([batch_src_caws_vec,batch_src_caws_ts_vec],dim=-1)
        batch_dst_caws_vec=torch.cat([batch_dst_caws_vec,batch_dst_caws_ts_vec],dim=-1)

        # Treat each walk as an independent sequence: [B*n_walk,walk_len+1,latent_dim+time_dim]
        batch_src_rnn_output,_=self.rnn_encoder(
            batch_src_caws_vec.reshape(-1,self.walk_len+1,self.latent_dim+self.time_dim)
        )
        batch_dst_rnn_output,_=self.rnn_encoder(
            batch_dst_caws_vec.reshape(-1,self.walk_len+1,self.latent_dim+self.time_dim)
        )

        # Padding 부분 처리
        batch_src_walk_lengths=batch_src_caws_mask.sum(dim=-1).reshape(-1)
        batch_dst_walk_lengths=batch_dst_caws_mask.sum(dim=-1).reshape(-1)
        walk_indices=torch.arange(src.size(0)*self.n_walk,device=device)
        batch_src_walk_vec=batch_src_rnn_output[
            walk_indices,batch_src_walk_lengths.clamp(min=1)-1
        ].reshape(src.size(0),self.n_walk,self.embed_dim)
        batch_dst_walk_vec=batch_dst_rnn_output[
            walk_indices,batch_dst_walk_lengths.clamp(min=1)-1
        ].reshape(src.size(0),self.n_walk,self.embed_dim)
        batch_src_walk_vec=batch_src_walk_vec.masked_fill(~batch_src_caws_mask.any(dim=-1,keepdim=True),0) # [B,n_walk,embed_dim]
        batch_dst_walk_vec=batch_dst_walk_vec.masked_fill(~batch_dst_caws_mask.any(dim=-1,keepdim=True),0) # [B,n_walk,embed_dim]

        """
        4. Aggregation
        """
        batch_walk_vec=torch.cat([batch_src_walk_vec,batch_dst_walk_vec],dim=1) # [B,2*n_walk,embed_dim]
        batch_walk_vec=self.walk_attn(batch_walk_vec) # [B,2*n_walk,embed_dim]
        pair_vec=batch_walk_vec.mean(dim=1) # [B,embed_dim]
        return pair_vec

class CAW_N_LP(CAW_N):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            embed_dim:int,
            n_head:int,
            n_walk:int,
            walk_len:int,
            alpha:float=0.0
        ):
        super(CAW_N_LP,self).__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            time_dim=time_dim,
            embed_dim=embed_dim,
            n_head=n_head,
            n_walk=n_walk,
            walk_len=walk_len,
            alpha=alpha
        )

        # CAW_N decoder for link prediction
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=embed_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=1
            )
        )

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor,
        ):
        pair_vec=self.embedding(
            src=src,
            dst=dst,
            event_t=event_t
        ) # [B,embed_dim]
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]