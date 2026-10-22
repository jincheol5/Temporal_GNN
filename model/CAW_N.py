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
            n_layer:int,
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
        self.n_layer=n_layer
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
        self.lstm_encoder=nn.LSTM(
            input_size=latent_dim+time_dim,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def set_graph(self,graph:CAW_N_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        batch_src_caws,batch_dst_caws=self.graph.set_based_anonymization(
            src=src,
            dst=dst,
            event_t=event_t
        )
        """
        
        """


        """
        1. Encode each anonymizated_vec_set tensor and timespan
        """


        """
        2. Encode walk using RNN
        """


