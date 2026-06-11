import torch
import pandas as pd
import numpy as np
from collections import defaultdict

class TemporalGraph:
    """
    node_id=0: padding node
    """
    def __init__(self,
            df:pd.DataFrame,
            node_dim:int
        ):
        """
        Input:
            df: pd.DataFrame, sorted by event time
            node_dim: int
        """
        # set adj, adj_t
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        for event in df.itertuples(index=False):
            src=int(event.src)
            dst=int(event.dst)
            t=int(event.t)
            edge_id=int(event.edge_id)
            self.adj[dst].append((src,edge_id))
            self.adj_t[dst].append(t)

        # set node_ft
        self.node_dim=node_dim
        self.num_node=max(df["src"].max(),df["dst"].max())
        self.node_ft=torch.zeros(
            (self.num_node+1,node_dim),
            dtype=torch.float32
        )

    def set_graph(self,
            df:pd.DataFrame,
            node_dim:int
        ):
        """
        Input:
            df: pd.DataFrame, sorted by event time
            node_dim: int
        """
        # set adj, adj_t
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        for event in df.itertuples(index=False):
            src=int(event.src)
            dst=int(event.dst)
            t=int(event.t)
            edge_id=int(event.edge_id)
            self.adj[dst].append((src,edge_id))
            self.adj_t[dst].append(t)

        # set node_ft
        self.node_dim=node_dim
        self.num_node=max(df["src"].max(),df["dst"].max())
        self.node_ft=torch.zeros(
            (self.num_node+1,node_dim),
            dtype=torch.float32
        )

    def get_num_node(self):
        return self.num_node

    def get_node_ft(self,node:torch.Tensor=None):
        """
        Input:
            node: [B,]
        Return:
            node_ft
        """
        device=node.device
        if node==None:
            return self.node_ft.to(device=device)
        else:
            return self.node_ft.to(device=device)[node]
    
    def get_temporal_neighbor(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            num_neighbor:int
        ):
        """
        Input:
            tar: [B,]
            tar_t: [B,]
            num_neighbor: int
            seed: int
        Return:
            neighbor_ids: [B,num_neighbor]
            neighbor_times: [B,num_neighbor]
            edge_ids: [B,num_neighbor]
        """
        device=tar.device
        batch_size=tar.size(0)
        neighbor_ids=torch.zeros(
            (batch_size,num_neighbor),
            dtype=torch.long,
            device=device
        )
        neighbor_times=torch.zeros(
            (batch_size,num_neighbor),
            dtype=torch.long,
            device=device
        )
        edge_ids=torch.zeros(
            (batch_size,num_neighbor),
            dtype=torch.long,
            device=device
        )
        for b in batch_size:
            tar_id=int(tar[b].item())
            cut_time=int(tar_t[b].item())

            neighbors=self.adj.get(tar_id,[])
            times=self.adj_t.get(tar_id,[])

            if len(neighbors)==0:
                continue

            # t < cut_time 인 마지막 위치까지 선택
            times_np=np.asarray(times,dtype=np.int64)
            cut_idx=np.searchsorted(
                times_np,
                cut_time,
                side="left"
            )

            # 최근 num_neighbor개만 선택
            start_idx=max(0,cut_idx-num_neighbor)
            selected_neighbors=neighbors[start_idx:cut_idx]
            selected_times=times[start_idx:cut_idx]

            # 앞은 0 padding, 뒤에 실제 neighbor 저장
            offset=num_neighbor-len(selected_neighbors)
            for idx,((src,edge_id),t) in enumerate(zip(selected_neighbors,selected_times)):
                neighbor_ids[b,offset+idx]=src
                neighbor_times[b,offset+idx]=t
                edge_ids[b,offset+idx]=edge_id
        return neighbor_ids,neighbor_times,edge_ids