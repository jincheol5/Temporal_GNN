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
            t=float(event.t)
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
            n_neighbor:int
        ):
        """
        Input:
            tar: [B,]
            tar_t: [B,]
            num_neighbor: int
            seed: int
        Return:
            neighbor_i: [B,num_neighbor]
            neighbor_t: [B,num_neighbor]
            neighbor_ts: [B,num_neighbor]
            edge_ids: [B,num_neighbor]
        """
        device=tar.device
        batch_size=tar.size(0)
        neighbor_id=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.long,
            device=device
        )
        neighbor_t=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.float,
            device=device
        )
        neighbor_ts=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.float,
            device=device
        )
        edge_id=torch.zeros(
            (batch_size,n_neighbor),
            dtype=torch.long,
            device=device
        )
        for b in range(batch_size):
            tar_id=int(tar[b].item())
            cut_time=float(tar_t[b].item())

            neighbors=self.adj.get(tar_id,[])
            times=self.adj_t.get(tar_id,[])

            if len(neighbors)==0:
                continue

            # t < cut_time 인 마지막 위치까지 선택
            times_np=np.asarray(times,dtype=np.float32)
            cut_idx=np.searchsorted(
                times_np,
                cut_time,
                side="left"
            )

            # 최근 num_neighbor개만 선택
            start_idx=max(0,cut_idx-n_neighbor)
            selected_neighbors=neighbors[start_idx:cut_idx]
            selected_times=times[start_idx:cut_idx]

            # 앞은 0 padding, 뒤에 실제 neighbor 저장
            offset=n_neighbor-len(selected_neighbors)
            for idx,((src,e_id),t) in enumerate(zip(selected_neighbors,selected_times)):
                neighbor_id[b,offset+idx]=src
                neighbor_t[b,offset+idx]=t
                neighbor_ts[b,offset+idx]=abs(cut_time-t)
                edge_id[b,offset+idx]=e_id
        return neighbor_id,neighbor_t,neighbor_ts,edge_id