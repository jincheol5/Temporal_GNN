import torch 
import torch.nn as nn
from graph import GNN_Graph

class TPNet_Matrix(nn.Module):
    def __init__(self,
            n_node:int,
            random_ft_dim:int=32,
            walk_len:int=5,
            time_decay_w:float=0.0001
        ):
        """

        """
        super().__init__()
        self.n_node=n_node
        self.random_ft_dim=random_ft_dim
        self.walk_len=walk_len
        self.time_decay_w=time_decay_w
        self.register_buffer(
            "random_ft",
            torch.zeros(
                (self.walk_len+1,self.n_node+1,self.random_ft_dim)
            )
        )
        self.init_random_ft()

    def init_random_ft(self):
        """
        NRM representation 초기화.

        random_ft: [walk_len+1, n_node+1, random_ft_dim]
            random_ft[0]: 실제 노드의 0-hop representation, N(0, 1/random_ft_dim).
            random_ft[1:]: 1~walk_len-hop representation, zero vector.
            random_ft[:,0,:]: 모든 hop의 padding node (node_id=0), zero vector.
        """
        self.random_ft.zero_()
        self.random_ft[0,1:].normal_(
            mean=0.0,
            std=self.random_ft_dim**-0.5
        )
        self.random_ft[:,0,:].zero_()

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def random_ft_propagation(self,

        ):
        """
        """